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
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from tqdm import tqdm

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ssl_dataset import IVFSSLDataset
from src.losses.stage_loss import stage_loss
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
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataloader(cfg: Dict) -> DataLoader:
    img_size = cfg["data"]["img_size"]
    tfm = build_transforms(img_size)
    dataset = IVFSSLDataset(
        csv_paths=[Path(p) for p in cfg["data"]["csv_paths"]],
        root_dir=Path(cfg["data"]["root_dir"]),
        transform1=tfm,
        transform2=tfm,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
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
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["ssl"].get("fp16", False))

    out_dir = Path(cfg["logging"]["out_dir"])
    logger = Logger(out_dir / "train.log")

    num_epochs = cfg["ssl"]["epochs"]
    lambda_stage = cfg["stage"]["lambda_stage"] if cfg["stage"]["enabled"] else 0.0
    num_stage_pos = cfg["stage"]["num_stage_positives"]
    log_every = cfg["logging"].get("log_every", 50)
    save_every = cfg["logging"].get("save_every_epochs", 10)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_byol = 0.0
        total_stage = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
        for step, (x1, x2, stage_ids, _) in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg["ssl"].get("fp16", False)):
                p1, p2, t1, t2 = model(x1, x2)
                loss_byol = byol_loss(p1, p2, t1, t2)
                # Stage loss uses target embeddings t1
                stage_l = stage_loss(t1.detach(), stage_ids, num_stage_pos) if lambda_stage > 0 else torch.tensor(
                    0.0, device=device
                )
                loss = loss_byol + lambda_stage * stage_l

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_moving_average()

            total_loss += loss.item()
            total_byol += loss_byol.item()
            total_stage += stage_l.item()

            if (step + 1) % log_every == 0:
                pbar.set_postfix(
                    loss=total_loss / (step + 1),
                    byol=total_byol / (step + 1),
                    stage=total_stage / (step + 1),
                )

        logger.log(
            f"Epoch {epoch}: loss={total_loss/len(loader):.4f}, byol={total_byol/len(loader):.4f}, stage={total_stage/len(loader):.4f}"
        )

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
