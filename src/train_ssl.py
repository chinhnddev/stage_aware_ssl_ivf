"""
Stage-aware BYOL pretraining entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from tqdm import tqdm

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ssl_dataset import IVFSSLDataset
from src.models.backbone import build_backbone
from src.models.byol import BYOL, byol_loss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import Logger
from src.utils.seed import set_seed


def build_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 0.3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataloader(cfg: Dict) -> DataLoader:
    data_cfg = cfg["data"]
    img_size = data_cfg["img_size"]
    tfm = build_transforms(img_size)
    dataset = IVFSSLDataset(
        csv_paths=[Path(p) for p in data_cfg["csv_paths"]],
        transform1=tfm,
        transform2=tfm,
        root_dir=Path(data_cfg["root_dir"]) if data_cfg.get("root_dir") else None,
        root_map=data_cfg.get("root_map"),
        use_domains=data_cfg.get("use_domains"),
        num_stage_positives=cfg["stage"]["num_stage_positives"],
    )
    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    return loader


def train(cfg: Dict) -> None:
    set_seed(cfg["ssl"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone, feat_dim = build_backbone(cfg["ssl"]["backbone"])
    model = BYOL(
        backbone=backbone,
        feature_dim=feat_dim,
        proj_dim=cfg["ssl"]["proj_dim"],
        pred_dim=cfg["ssl"]["pred_dim"],
        tau=cfg["ssl"]["ema_tau"],
    ).to(device)

    loader = create_dataloader(cfg)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["ssl"]["lr"], weight_decay=cfg["ssl"]["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["ssl"].get("fp16", False))

    out_dir = Path(cfg["logging"]["out_dir"])
    logger = Logger(out_dir / "train.log")

    num_epochs = cfg["ssl"]["epochs"]
    lambda_stage = cfg["stage"]["lambda_stage"] if cfg["stage"]["enabled"] else 0.0
    log_every = cfg["logging"].get("log_every", 50)
    save_every = cfg["logging"].get("save_every_epochs", 10)
    patience = cfg["logging"].get("early_stop_patience")
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_byol = 0.0
        total_stage = 0.0
        total_stage_pos_anchors = 0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
        dataset = loader.dataset
        for step, (x1, x2, stage_ids, pos_lists, _) in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg["ssl"].get("fp16", False)):
                p1, p2, t1, t2 = model(x1, x2)
                loss_byol = byol_loss(p1, p2, t1, t2)
                if lambda_stage > 0:
                    stage_terms = []
                    stage_pos_anchors = 0
                    # Collect unique positive indices across the batch
                    all_pos_indices = {pi for pis in pos_lists for pi in pis}
                    pos_emb_dict: Dict[int, torch.Tensor] = {}
                    if all_pos_indices:
                        with torch.no_grad():
                            pos_imgs = torch.stack([dataset.get_view(pi, view=1) for pi in all_pos_indices]).to(
                                device, non_blocking=True
                            )
                            t_pos = model.target_projector(model.target_backbone(pos_imgs))
                            t_pos = F.normalize(t_pos, dim=-1)
                            pos_emb_dict = {pi: emb for pi, emb in zip(all_pos_indices, t_pos)}

                    anchor_t = F.normalize(t1.detach(), dim=-1)
                    for anchor_emb, pos_idx_list in zip(anchor_t, pos_lists):
                        valid = [pi for pi in pos_idx_list if pi in pos_emb_dict]
                        if not valid:
                            continue
                        stage_pos_anchors += 1
                        for pi in valid:
                            stage_terms.append(1 - torch.dot(anchor_emb, pos_emb_dict[pi]))

                    stage_l = torch.stack(stage_terms).mean() if stage_terms else torch.tensor(0.0, device=device)
                else:
                    stage_l = torch.tensor(0.0, device=device)
                    stage_pos_anchors = 0

                loss = loss_byol + lambda_stage * stage_l

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_moving_average()

            total_loss += loss.item()
            total_byol += loss_byol.item()
            total_stage += stage_l.item()
            total_stage_pos_anchors += stage_pos_anchors

            if (step + 1) % log_every == 0:
                pbar.set_postfix(
                    loss=total_loss / (step + 1),
                    byol=total_byol / (step + 1),
                    stage=total_stage / (step + 1),
                    stage_pos=total_stage_pos_anchors / (step + 1),
                )

        epoch_loss = total_loss / len(loader)
        logger.log(
            f"Epoch {epoch}: loss={epoch_loss:.4f}, byol={total_byol/len(loader):.4f}, stage={total_stage/len(loader):.4f}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
            }
            save_checkpoint(best_ckpt, out_dir / "best.ckpt")
            logger.log(f"New best loss {best_loss:.4f} at epoch {epoch}, saved best.ckpt")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                logger.log(
                    f"Early stopping at epoch {epoch} (no improvement for {patience} epochs; best loss {best_loss:.4f})"
                )
                break

        if epoch % save_every == 0 or epoch == num_epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg,
            }
            save_checkpoint(ckpt, out_dir / f"epoch_{epoch}.ckpt")

    # Save last
    ckpt = {
        "epoch": num_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }
    save_checkpoint(ckpt, out_dir / "last.ckpt")
    # Save backbone only
    save_checkpoint(model.online_backbone.state_dict(), out_dir / "encoder_ssl.pth")
    logger.log(f"Saved encoder to {out_dir/'encoder_ssl.pth'}")
    logger.close()


def export_encoder(ckpt_path: Path, out_path: Path) -> None:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # Expect BYOL full state; extract backbone weights
    backbone_state = {k.replace("online_backbone.", ""): v for k, v in state.items() if k.startswith("online_backbone.")}
    save_checkpoint(backbone_state, out_path)
    print(f"Exported encoder weights to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-aware BYOL pretraining.")
    parser.add_argument("--config", type=Path, help="Path to YAML config.")
    parser.add_argument("--export_encoder", type=Path, help="Checkpoint to export backbone from.")
    parser.add_argument("--out", type=Path, help="Output path for exported encoder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_encoder and args.out:
        export_encoder(args.export_encoder, args.out)
        return
    if not args.config:
        raise ValueError("Config path is required for training.")
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
