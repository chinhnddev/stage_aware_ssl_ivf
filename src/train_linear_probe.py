"""
WP2 linear probing baseline with frozen backbone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Local imports
from src.data.supervised_dataset import IVFClassifDataset
from src.data.supervised_dataloader import _make_transforms
from src.models.classifier import LinearHead
from src.models.frozen_backbone import build_frozen_backbone, FrozenBackboneEncoder
from src.utils.metrics import compute_classification_metrics, find_best_threshold
from src.utils.seed import set_seed


def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    img_size = data_cfg["img_size"]
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)
    root_map = data_cfg["root_map"]
    sup_csv = Path(data_cfg["supervised_csv"])
    cross_csv = Path(data_cfg["cross_csv"])

    train_tf = _make_transforms(img_size, train=True)
    eval_tf = _make_transforms(img_size, train=False)

    def make_loader(split: str, csv_path: Path, domain: str, role: str, tf, shuffle: bool) -> DataLoader:
        ds = IVFClassifDataset(
            csv_path=csv_path,
            split=split,
            root_map=root_map,
            use_domain=domain,
            use_role=role,
            transform=tf,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = make_loader("train", sup_csv, "hv_kaggle", "supervised", train_tf, shuffle=True)
    val_loader = make_loader("val", sup_csv, "hv_kaggle", "supervised", eval_tf, shuffle=False)
    test_loader = make_loader("test", sup_csv, "hv_kaggle", "supervised", eval_tf, shuffle=False)
    # Clinical split might be "test" or "holdout"; try test first then fall back.
    try:
        clinical_loader = make_loader("test", cross_csv, "hv_clinical", "crossdomain_test", eval_tf, shuffle=False)
    except ValueError:
        clinical_loader = make_loader("holdout", cross_csv, "hv_clinical", "crossdomain_test", eval_tf, shuffle=False)
    return train_loader, val_loader, test_loader, clinical_loader


def build_model(cfg: Dict, device: torch.device) -> Tuple[FrozenBackboneEncoder, nn.Module]:
    mcfg = cfg["model"]
    mode = mcfg.get("mode", "imagenet").lower()
    backbone_name = mcfg.get("backbone_name", "resnet50")
    ssl_ckpt = mcfg.get("ssl_checkpoint") or mcfg.get("checkpoint") or ""
    if mode == "imagenet":
        backbone = build_frozen_backbone(
            name=backbone_name,
            pretrained=True,  # use ImageNet weights
            checkpoint=None,
            from_byol_ckpt=False,
        ).to(device)
    elif mode == "ssl":
        if not ssl_ckpt:
            raise ValueError("mode=ssl requires model.ssl_checkpoint to be set.")
        ckpt_path = Path(ssl_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SSL checkpoint not found: {ckpt_path}")
        # pretrained=False to avoid mixing ImageNet and SSL weights
        backbone = build_frozen_backbone(
            name=backbone_name,
            pretrained=False,
            checkpoint=ckpt_path,
            from_byol_ckpt=mcfg.get("from_byol_ckpt", True),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model.mode='{mode}' (expected imagenet or ssl)")
    print(f"[model] mode={mode} backbone={backbone_name} ssl_checkpoint={ssl_ckpt or 'None'}")
    head = LinearHead(backbone.feature_dim).to(device)
    return backbone, head


def eval_model(backbone: FrozenBackboneEncoder, head: nn.Module, loader: DataLoader, device: torch.device, threshold: float):
    backbone.eval()
    head.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)  # ensure 1D labels
            feats = backbone(x)
            logits = head(feats).view(-1)  # ensure 1D logits
            y_true.append(y.cpu())
            y_score.append(torch.sigmoid(logits).cpu())
    y_true_np = torch.cat(y_true).numpy()
    y_score_np = torch.cat(y_score).numpy()
    return compute_classification_metrics(y_true_np, y_score_np, threshold=threshold)


def collect_scores(backbone: FrozenBackboneEncoder, head: nn.Module, loader: DataLoader, device: torch.device):
    backbone.eval()
    head.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)  # ensure 1D labels
            feats = backbone(x)
            logits = head(feats).view(-1)  # ensure 1D logits
            y_true.append(y.cpu())
            y_score.append(torch.sigmoid(logits).cpu())
    return torch.cat(y_true).numpy(), torch.cat(y_score).numpy()


def train(cfg: Dict) -> None:
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, clinical_loader = build_loaders(cfg)
    backbone, head = build_model(cfg, device)

    # Sanity on trainable params
    head_params = [p for p in head.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in head_params)
    frozen_params = sum(p.numel() for p in backbone.parameters())
    print(f"[model] frozen params: {frozen_params:,} | trainable head params: {total_trainable:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(head.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"].get("fp16", False))

    run_name = cfg.get("logging", {}).get("run_name", "wp2_linear_probe")
    mode = cfg["model"].get("mode", "imagenet")
    out_dir_base = Path(cfg["logging"]["out_dir"])
    out_dir = out_dir_base / f"{run_name}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_state = None

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        head.train()
        backbone.eval()  # keep frozen
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            with torch.amp.autocast("cuda", enabled=cfg["train"].get("fp16", False)):
                feats = backbone(x)
                logits = head(feats)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Val
        y_true_np, y_score_np = collect_scores(backbone, head, val_loader, device)
        best_thresh, best_f1 = find_best_threshold(y_true_np, y_score_np)
        val_metrics = compute_classification_metrics(y_true_np, y_score_np, threshold=best_thresh)
        if val_metrics["auprc"] > best_val:
            best_val = val_metrics["auprc"]
            best_state = {
                "epoch": epoch,
                "head": head.state_dict(),
                "cfg": cfg,
                "threshold": best_thresh,
                "val_metrics": val_metrics,
            }
            torch.save(best_state, out_dir / "best_linear_probe.pth")
        print(
            f"[epoch {epoch}] loss={avg_loss:.4f} val_auprc={val_metrics['auprc']:.4f} val_f1={val_metrics['f1']:.4f} best_thresh={best_thresh:.3f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished without any validation metrics recorded.")
    head.load_state_dict(best_state["head"])
    threshold = best_state["threshold"]

    test_metrics = eval_model(backbone, head, test_loader, device, threshold)
    clinical_metrics = eval_model(backbone, head, clinical_loader, device, threshold)

    summary = {
        "config": {
            "backbone": cfg["model"]["backbone_name"],
            "mode": cfg["model"].get("mode", "imagenet"),
            "checkpoint": cfg["model"].get("checkpoint", ""),
            "ssl_checkpoint": cfg["model"].get("ssl_checkpoint", ""),
            "from_byol_ckpt": cfg["model"].get("from_byol_ckpt", False),
            "pretrained": cfg["model"].get("pretrained", True),
            "img_size": cfg["data"]["img_size"],
            "supervised_csv": str(cfg["data"]["supervised_csv"]),
            "cross_csv": str(cfg["data"]["cross_csv"]),
        },
        "threshold": threshold,
        "val_metrics": best_state["val_metrics"],
        "test_kaggle": test_metrics,
        "test_clinical": clinical_metrics,
        "trainable_params": total_trainable,
        "frozen_params": frozen_params,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WP2 linear probe with frozen backbone.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
