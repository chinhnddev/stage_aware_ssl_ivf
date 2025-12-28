"""
Stage-aware BYOL pretraining entry point.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List
import json

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
from src.losses.stage_loss import stage_loss_from_pairs
from src.models.backbone import build_backbone
from src.models.byol import BYOL, byol_loss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import Logger
from src.utils.seed import set_seed


def export_encoder_ckpt(backbone: nn.Module, out_path: Path, meta: Dict) -> None:
    """
    Export backbone-only state dict plus a meta json alongside for downstream loading.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(backbone.state_dict(), out_path)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] saved encoder state to {out_path} and meta to {meta_path}")


def sanity_check_strict(backbone_name: str, encoder_path: Path) -> None:
    """Strict load to ensure exported encoder matches backbone definition."""
    model, _ = build_backbone(backbone_name, pretrained=False)
    state = torch.load(encoder_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
        print("[sanity] encoder_ssl.pth loads with strict=True")
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[sanity] strict load failed: {e}")
        print(f"  missing: {missing}")
        print(f"  unexpected: {unexpected}")
        raise


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
    use_roles = data_cfg.get("use_roles", ["ssl"])
    num_workers = data_cfg.get("num_workers", 2)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    # Warn on high worker count in Colab
    if str(Path.cwd()).startswith("/content") and num_workers > 4:
        print(f"[warn] num_workers={num_workers} may be high on Colab; consider 2–4.")
    # Warn if data on Drive for I/O slowness
    for p in data_cfg.get("csv_paths", []):
        if "/content/drive" in str(p):
            print("[warn] CSV under /content/drive; consider copying data to /content for faster I/O.")
            break
    dataset = IVFSSLDataset(
        csv_paths=[Path(p) for p in data_cfg["csv_paths"]],
        transform1=tfm,
        transform2=tfm,
        root_dir=Path(data_cfg["root_dir"]) if data_cfg.get("root_dir") else None,
        root_map=data_cfg.get("root_map"),
        use_domains=data_cfg.get("use_domains"),
        use_roles=use_roles,
        num_stage_positives=cfg["stage"]["num_stage_positives"],
    )
    # Note: shuffle=True with the default sampler; no stage-balanced sampler is implemented,
    # so stage.min_per_stage_in_batch (if set) is not enforced.
    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    print(
        f"[dataloader] samples={len(dataset)} batches={len(loader)} "
        f"batch_size={data_cfg['batch_size']} num_workers={data_cfg['num_workers']} "
        f"pin_memory={torch.cuda.is_available()} persistent_workers={data_cfg['num_workers']>0}"
    )
    return loader


def train(cfg: Dict) -> None:
    set_seed(cfg["ssl"].get("seed", 42))
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    fp16_enabled = cfg["ssl"].get("fp16", False) and cuda_available
    if not cuda_available and cfg["ssl"].get("fp16", False):
        print("[env] device=cpu cuda_available=False fp16=False (forced off)")
    else:
        print(f"[env] device={device} cuda_available={cuda_available} fp16={fp16_enabled}")

    backbone_name = cfg["ssl"]["backbone"]
    backbone_pretrained = cfg["ssl"].get("pretrained", False)
    backbone, feat_dim = build_backbone(backbone_name, pretrained=backbone_pretrained)
    model = BYOL(
        backbone=backbone,
        feature_dim=feat_dim,
        proj_dim=cfg["ssl"]["proj_dim"],
        pred_dim=cfg["ssl"]["pred_dim"],
        tau=cfg["ssl"]["ema_tau"],
    ).to(device)

    if cfg["stage"].get("enabled") and cfg["stage"].get("min_per_stage_in_batch", 0) > 0:
        warnings.warn(
            "stage.min_per_stage_in_batch is currently unused: DataLoader uses shuffle=True and no stage-balanced sampler is implemented.",
            stacklevel=2,
        )

    loader = create_dataloader(cfg)

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["ssl"]["lr"], weight_decay=cfg["ssl"]["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16_enabled)

    out_dir = Path(cfg["logging"]["out_dir"])
    logger = Logger(out_dir / "train.log")

    num_epochs = cfg["ssl"]["epochs"]
    lambda_stage = cfg["stage"]["lambda_stage"] if cfg["stage"]["enabled"] else 0.0
    log_every = cfg["logging"].get("log_every", 50)
    save_every = cfg["logging"].get("save_every_epochs", 10)
    export_every = cfg["logging"].get("export_encoder_every_epochs", save_every)
    patience = cfg["logging"].get("early_stop_patience")
    best_loss = float("inf")
    start_epoch = 1

    # Resume support
    resume_path = cfg.get("logging", {}).get("resume_from")
    if resume_path:
        rpath = Path(resume_path)
        if rpath.exists():
            ckpt_resume = load_checkpoint(rpath, map_location=device)
            model.load_state_dict(ckpt_resume["model"])
            optimizer.load_state_dict(ckpt_resume["optimizer"])
            start_epoch = int(ckpt_resume.get("epoch", 0)) + 1
            best_loss = ckpt_resume.get("best_loss", best_loss)
            print(f"[resume] Loaded {rpath}, starting at epoch {start_epoch}, best_loss={best_loss:.4f}")
        else:
            print(f"[warn] resume_from path not found: {rpath}; starting from scratch.")
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_byol = 0.0
        total_stage = 0.0
        total_stage_pos_anchors = 0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
        dataset = loader.dataset
        try:
            for step, (x1, x2, stage_ids, pos_lists, _) in pbar:
                # Normalize pos_lists to list of list[int] to avoid tensor membership issues
                if isinstance(pos_lists, torch.Tensor):
                    pos_lists = pos_lists.tolist()
                else:
                    pos_lists = [
                        pl.tolist() if isinstance(pl, torch.Tensor) else pl  # type: ignore[union-attr]
                        for pl in pos_lists
                    ]
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=fp16_enabled):
                    p1, p2, t1, t2 = model(x1, x2)
                    loss_byol = byol_loss(p1, p2, t1, t2)
                    if lambda_stage > 0:
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

                        anchor_z = p1
                        stage_l, stage_pos_anchors = stage_loss_from_pairs(anchor_z, pos_emb_dict, pos_lists)
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
        except KeyboardInterrupt:
            print("[info] Training interrupted by user.")
            break

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
                "best_loss": best_loss,
            }
            save_checkpoint(best_ckpt, out_dir / "best.ckpt")
            meta = {
                "backbone_name": backbone_name,
                "img_size": cfg["data"]["img_size"],
                "epoch_saved": epoch,
                "pretrained": backbone_pretrained,
                "global_pool": "avg",
                "num_classes": 0,
            }
            export_encoder_ckpt(model.online_backbone, out_dir / "encoder_ssl.pth", meta)
            sanity_check_strict(backbone_name, out_dir / "encoder_ssl.pth")
            logger.log(f"New best loss {best_loss:.4f} at epoch {epoch}, saved best.ckpt and encoder_ssl.pth")
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
                "best_loss": best_loss,
            }
            save_checkpoint(ckpt, out_dir / f"epoch_{epoch}.ckpt")
            meta = {
                "backbone_name": backbone_name,
                "img_size": cfg["data"]["img_size"],
                "epoch_saved": epoch,
                "pretrained": backbone_pretrained,
                "global_pool": "avg",
                "num_classes": 0,
            }
            export_encoder_ckpt(model.online_backbone, out_dir / "encoder_ssl.pth", meta)
            sanity_check_strict(backbone_name, out_dir / "encoder_ssl.pth")
        if epoch % export_every == 0:
            meta = {
                "backbone_name": backbone_name,
                "img_size": cfg["data"]["img_size"],
                "epoch_saved": epoch,
                "pretrained": backbone_pretrained,
                "global_pool": "avg",
                "num_classes": 0,
            }
            export_encoder_ckpt(model.online_backbone, out_dir / "encoder_ssl.pth", meta)
            sanity_check_strict(backbone_name, out_dir / "encoder_ssl.pth")

    # Save last
    ckpt = {
        "epoch": num_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "best_loss": best_loss,
    }
    save_checkpoint(ckpt, out_dir / "last.ckpt")
    # Save backbone only with meta
    meta = {
        "backbone_name": backbone_name,
        "img_size": cfg["data"]["img_size"],
        "epoch_saved": num_epochs,
        "pretrained": backbone_pretrained,
        "global_pool": "avg",
        "num_classes": 0,
    }
    export_encoder_ckpt(model.online_backbone, out_dir / "encoder_ssl.pth", meta)
    sanity_check_strict(backbone_name, out_dir / "encoder_ssl.pth")
    logger.log(f"Saved encoder to {out_dir/'encoder_ssl.pth'}")
    logger.close()


def export_encoder(ckpt_path: Path, out_path: Path) -> None:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # Expect BYOL full state; extract backbone weights
    backbone_state = {k.replace("online_backbone.", ""): v for k, v in state.items() if k.startswith("online_backbone.")}
    save_checkpoint(backbone_state, out_path)
    print(f"Exported encoder weights to {out_path}")


def sanity_check_encoder(cfg: Dict, encoder_path: Path) -> None:
    """Load backbone-only checkpoint with strict=True to verify compatibility."""
    backbone_name = cfg["ssl"]["backbone"]
    model, _ = build_backbone(backbone_name, pretrained=False)
    state = torch.load(encoder_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    print("Encoder checkpoint loads successfully with strict=True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-aware BYOL pretraining.")
    parser.add_argument("--config", type=Path, help="Path to YAML config.")
    parser.add_argument("--export_encoder", type=Path, help="Checkpoint to export backbone from.")
    parser.add_argument("--out", type=Path, help="Output path for exported encoder.")
    parser.add_argument("--sanity_check_encoder", action="store_true", help="Run strict load on encoder_ssl.pth and exit.")
    parser.add_argument("--encoder_path", type=Path, help="Path to encoder_ssl.pth (defaults to config logging.out_dir/encoder_ssl.pth).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_encoder and args.out:
        export_encoder(args.export_encoder, args.out)
        return
    if not args.config:
        raise ValueError("Config path is required for training.")
    cfg = load_config(args.config)
    if args.sanity_check_encoder:
        enc_path = args.encoder_path or (Path(cfg["logging"]["out_dir"]) / "encoder_ssl.pth")
        sanity_check_encoder(cfg, enc_path)
        return
    train(cfg)


if __name__ == "__main__":
    main()
