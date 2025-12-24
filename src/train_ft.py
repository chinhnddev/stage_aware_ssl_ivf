"""
Supervised finetuning using pretrained encoder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import sys

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.supervised_dataloader import create_supervised_loaders
from src.models.classifier import (
    EmbryoClassifier,
    LinearHead,
    MLPHead,
    build_backbone,
    freeze_backbone,
    load_pretrained_backbone,
)
from src.utils.metrics import compute_classification_metrics, find_best_threshold
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint


def build_model(cfg: Dict, device: torch.device) -> EmbryoClassifier:
    backbone, feat_dim = build_backbone(cfg["model"]["backbone"])
    head_type = cfg["model"].get("head", "mlp")
    if head_type == "linear":
        head = LinearHead(feat_dim)
    else:
        head = MLPHead(feat_dim, dropout=cfg["model"].get("dropout", 0.2))
    model = EmbryoClassifier(backbone, head).to(device)
    if cfg["model"]["pretrained"].get("enabled", False):
        load_pretrained_backbone(
            model.backbone,
            Path(cfg["model"]["pretrained"]["path"]),
            from_byol_ckpt=cfg["model"]["pretrained"].get("from_byol_ckpt", False),
        )
    return model


def compute_pos_weight(loader: DataLoader) -> torch.Tensor:
    labels = []
    for _, y in loader:
        labels.append(y)
    y_all = torch.cat(labels)
    pos = (y_all == 1).sum().item()
    neg = (y_all == 0).sum().item()
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(pos, 1))


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> Dict:
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).view(-1)
            y_true.append(y.cpu())
            y_score.append(torch.sigmoid(logits).cpu())
    y_true_np = torch.cat(y_true).numpy()
    y_score_np = torch.cat(y_score).numpy()
    return compute_classification_metrics(y_true_np, y_score_np, threshold=threshold)


def collect_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).view(-1)
            y_true.append(y.cpu())
            y_score.append(torch.sigmoid(logits).cpu())
    return torch.cat(y_true).numpy(), torch.cat(y_score).numpy()


def train(cfg: Dict) -> None:
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "_prebuilt_loaders" in cfg:
        train_loader, val_loader, test_loader = cfg["_prebuilt_loaders"]
    else:
        train_loader, val_loader, test_loader = create_supervised_loaders(cfg)

    model = build_model(cfg, device)

    pos_weight_cfg = cfg["train"].get("pos_weight", "auto")
    if pos_weight_cfg == "auto":
        pos_weight = compute_pos_weight(train_loader).to(device)
    else:
        pos_weight = torch.tensor(float(pos_weight_cfg), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head_params = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters()]

    def make_optim(unfreeze_backbone: bool):
        params = [{"params": head_params, "lr": cfg["train"]["lr_head"], "weight_decay": cfg["train"]["weight_decay"]}]
        if unfreeze_backbone:
            params.append(
                {"params": backbone_params, "lr": cfg["train"]["lr_backbone"], "weight_decay": cfg["train"]["weight_decay"]}
            )
        return optim.AdamW(params)

    freeze_backbone(model.backbone, until="none")  # fully frozen
    optimizer = make_optim(unfreeze_backbone=False)

    scheduler_name = cfg["train"].get("scheduler", "cosine")
    scheduler_key = scheduler_name.lower()
    if scheduler_key == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    elif scheduler_key in ("plateau", "reduce", "reducelronplateau"):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("fp16", False))

    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"
    log_f = open(log_file, "a", encoding="utf-8")

    best_metric = -1.0
    patience_ctr = 0
    threshold_cfg = cfg["eval"].get("threshold", 0.5)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)
            with torch.cuda.amp.autocast(enabled=cfg["train"].get("fp16", False)):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        # Eval on val (probabilities; no threshold search inside loop)
        val_metrics = eval_model(model, val_loader, device, threshold_cfg)
        val_score = val_metrics["auprc"]

        if scheduler_key == "cosine" and scheduler is not None:
            scheduler.step()
        elif scheduler_key in ("plateau", "reduce", "reducelronplateau") and scheduler is not None:
            scheduler.step(val_score)

        # Early stop / checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
            "pos_weight": pos_weight.item(),
        }
        save_checkpoint(ckpt, out_dir / "last.ckpt")
        if val_score > best_metric:
            best_metric = val_score
            patience_ctr = 0
            save_checkpoint(ckpt, out_dir / "best_val_auprc.ckpt")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["train"].get("early_stop_patience", 10):
                print(f"[early stop] no improvement for {patience_ctr} epochs; stopping.")
                break

        log_f.write(
            f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.4f}, val_auprc={val_metrics['auprc']:.4f}, val_auroc={val_metrics['auroc']:.4f}\n"
        )
        log_f.flush()

        # Unfreeze after warmup
        if epoch == cfg["train"]["warmup_epochs"]:
            freeze_backbone(model.backbone, until=cfg["train"].get("unfreeze", "layer4"))
            optimizer = make_optim(unfreeze_backbone=True)
            if scheduler_key == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"] - epoch)
            elif scheduler_key in ("plateau", "reduce", "reducelronplateau"):
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
            else:
                scheduler = None

    # Final eval with best checkpoint using the same splits/loaders
    best_ckpt_path = out_dir / "best_val_auprc.ckpt"
    if best_ckpt_path.exists():
        state = load_checkpoint(best_ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    if cfg["eval"].get("tune_threshold_on_val", True):
        y_true_np, y_score_np = collect_scores(model, val_loader, device)
        best_thresh, _ = find_best_threshold(y_true_np, y_score_np)
    else:
        best_thresh = cfg["eval"].get("threshold", 0.5)

    val_metrics = eval_model(model, val_loader, device, threshold=best_thresh)
    test_metrics = eval_model(model, test_loader, device, threshold=best_thresh)
    metrics_out = {"val": val_metrics, "test": test_metrics, "threshold": best_thresh}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print(json.dumps(metrics_out, indent=2))
    log_f.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune embryo classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt", type=Path, help="Checkpoint to eval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    # Ensure split persistence: create loaders once so split file is reused
    train_loader, val_loader, test_loader = create_supervised_loaders(cfg)
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, device)
        state = load_checkpoint(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        if cfg["eval"].get("tune_threshold_on_val", True):
            y_true_np, y_score_np = collect_scores(model, val_loader, device)
            best_thresh, _ = find_best_threshold(y_true_np, y_score_np)
        else:
            best_thresh = cfg["eval"].get("threshold", 0.5)
        val_metrics = eval_model(model, val_loader, device, threshold=best_thresh)
        test_metrics = eval_model(model, test_loader, device, threshold=best_thresh)
        metrics_out = {"val": val_metrics, "test": test_metrics, "threshold": best_thresh}
        print(json.dumps(metrics_out, indent=2))
        return
    # Pass prebuilt loaders through cfg to reuse
    cfg["_prebuilt_loaders"] = (train_loader, val_loader, test_loader)
    train(cfg)


if __name__ == "__main__":
    main()
