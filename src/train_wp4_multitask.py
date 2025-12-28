"""
WP4 training script: frozen backbone + WP3 features + quality/stage heads + calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

from src.models.frozen_backbone import build_frozen_backbone
from src.models.wp3_modules import WP3FeatureComposer
from src.models.wp4_heads import QualityHead, StageHead, compute_multitask_loss
from src.utils.metrics import compute_classification_metrics, find_best_threshold
from src.utils.calibration import TemperatureScaler, compute_ece, apply_temperature
from src.utils.seed import set_seed
from src.data.supervised_dataloader import _make_transforms
from src.utils.domain_alignment import coral_loss


class MultitaskDataset(Dataset):
    """Loads quality label, domain id, and optional stage label."""

    def __init__(
        self,
        csv_path: Path,
        split: Optional[str],
        root_map: Dict[str, str],
        use_domain: str,
        role: str,
        transform,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = df[(df["domain"] == use_domain) & (df["role"] == role)]
        if split:
            df = df[df["split"] == split]
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} domain={use_domain} role={role} in {csv_path}")
        self.transform = transform
        self.root = Path(root_map[use_domain])
        self.paths = df["image_path"].tolist()
        self.quality = df["quality_label"].astype(float).tolist()
        self.stage_raw = df["stage_raw"].tolist() if "stage_raw" in df.columns else [None] * len(df)
        self.stage_mask = pd.notna(df["stage_raw"]).astype(float).tolist() if "stage_raw" in df.columns else [0.0] * len(df)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.root / self.paths[idx]
        with open(img_path, "rb") as f:
            from PIL import Image
            img = Image.open(f).convert("RGB")
        img = self.transform(img)
        return (
            img,
            torch.tensor(self.quality[idx], dtype=torch.float32),
            self.stage_raw[idx],
            torch.tensor(self.stage_mask[idx], dtype=torch.float32),
        )


def build_loaders(cfg: Dict):
    data_cfg = cfg["data"]
    tf_train = _make_transforms(data_cfg["img_size"], train=True)
    tf_eval = _make_transforms(data_cfg["img_size"], train=False)
    root_map = data_cfg["root_map"]
    sup_csv = Path(data_cfg["supervised_csv"])
    cross_csv = Path(data_cfg["cross_csv"])
    target_unlabeled_csv = Path(data_cfg["target_unlabeled_csv"]) if data_cfg.get("target_unlabeled_csv") else None

    def make(csv_path: Path, split: str, use_domain: str, role: str, tf, shuffle: bool):
        ds = MultitaskDataset(csv_path, split, root_map, use_domain, role, tf)
        return DataLoader(ds, batch_size=data_cfg["batch_size"], shuffle=shuffle, num_workers=data_cfg["num_workers"], pin_memory=True)

    train_loader = make(sup_csv, "train", "hv_kaggle", "supervised", tf_train, True)
    val_loader = make(sup_csv, "val", "hv_kaggle", "supervised", tf_eval, False)
    test_loader = make(sup_csv, "test", "hv_kaggle", "supervised", tf_eval, False)
    clinical_loader = make(cross_csv, "test", "hv_clinical", "crossdomain_test", tf_eval, False)
    target_loader = None
    if target_unlabeled_csv and target_unlabeled_csv.exists():
        try:
            target_loader = make(target_unlabeled_csv, "unlabeled", "hv_clinical", "target_unlabeled", tf_train, True)
        except Exception:
            target_loader = None
    return train_loader, val_loader, test_loader, clinical_loader, target_loader


def encode_domain_ids(domain: str) -> int:
    vocab = {"roboflow_ssl": 0, "hv_kaggle": 1, "hv_clinical": 2}
    return vocab.get(domain, 0)


def run_eval(
    backbone,
    composer,
    q_head,
    s_head,
    loader,
    device,
    stage_classes: Dict[str, int],
    domain_str: str,
    temperature: Optional[float] = None,
):
    backbone.eval()
    composer.eval()
    q_head.eval()
    if s_head:
        s_head.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for imgs, q, stage_lbl, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            feats = backbone(imgs)
            domain_ids = torch.full((imgs.size(0),), encode_domain_ids(domain_str), device=device, dtype=torch.long)
            fused, aux = composer(feats, domain_ids=domain_ids, stage_labels=list(stage_lbl), align=False)
            logits_q = q_head(fused)
            if temperature:
                logits_q = logits_q / temperature
            y_true.append(q.cpu())
            y_score.append(torch.sigmoid(logits_q).cpu())
    y_true_np = torch.cat(y_true).numpy()
    y_score_np = torch.cat(y_score).numpy()
    return y_true_np, y_score_np


def train(cfg: Dict) -> None:
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, clinical_loader, target_loader = build_loaders(cfg)

    backbone = build_frozen_backbone(
        name=cfg["model"]["backbone_name"],
        pretrained=cfg["model"].get("pretrained", True),
        checkpoint=Path(cfg["model"]["checkpoint"]) if cfg["model"].get("checkpoint") else None,
        from_byol_ckpt=cfg["model"].get("from_byol_ckpt", False),
    ).to(device)

    composer = WP3FeatureComposer(
        in_dim=backbone.feature_dim,
        token_dim=cfg["wp3"]["token_dim"],
        adapter_cfg=cfg["wp3"]["adapter"],
        morph_cfg=cfg["wp3"]["morph"],
        gen_cfg=cfg["wp3"]["gen"],
    ).to(device)

    fused_dim = cfg["wp3"]["token_dim"] * 2
    q_head = QualityHead(fused_dim, hidden=cfg["wp4"]["quality_head"].get("hidden"), dropout=cfg["wp4"]["quality_head"].get("dropout", 0.2)).to(device)
    stage_classes = cfg["wp4"]["stage_classes"]
    s_head = StageHead(fused_dim, num_classes=len(stage_classes)).to(device) if cfg["wp4"]["use_stage_head"] else None

    params = list(q_head.parameters()) + list(composer.parameters())
    if s_head:
        params += list(s_head.parameters())
    optimizer = optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"].get("fp16", False))

    pos_weight = torch.tensor(cfg["wp4"]["quality_head"].get("pos_weight", 1.0), device=device)

    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_state = None

    domain_align_mode = cfg["model"].get("domain_alignment", "grl").lower()
    lambda_domain = cfg["model"].get("lambda_domain_align", 0.0)
    target_iter = iter(target_loader) if target_loader is not None else None

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        composer.train()
        q_head.train()
        if s_head:
            s_head.train()
        total_loss = 0.0
        total_coral = 0.0
        for imgs, q, stage_lbl, stage_mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            stage_mask = stage_mask.to(device, non_blocking=True)
            domain_ids = torch.full((imgs.size(0),), encode_domain_ids("hv_kaggle"), device=device, dtype=torch.long)

            with torch.amp.autocast("cuda", enabled=cfg["train"].get("fp16", False)):
                feats = backbone(imgs)
                fused, aux = composer(
                    feats,
                    domain_ids=domain_ids,
                    stage_labels=list(stage_lbl),
                    align=cfg["wp3"]["gen"].get("use_domain_align", False) and domain_align_mode == "grl",
                )
                logits_q = q_head(fused)
                logits_stage = s_head(fused) if s_head else None
                if logits_stage is not None:
                    stage_ids = torch.tensor([stage_classes.get(str(s).upper(), 0) for s in stage_lbl], device=device)
                else:
                    stage_ids = None
                losses = compute_multitask_loss(
                    logits_q,
                    q,
                    logits_stage,
                    stage_ids,
                    stage_mask if s_head else None,
                    lambda_stage=cfg["wp4"].get("lambda_stage", 0.0),
                    pos_weight=pos_weight,
                    focal_gamma=cfg["wp4"]["quality_head"].get("focal_gamma", 0.0),
                )
                loss = losses["total"]

                # CORAL alignment (training-only)
                if domain_align_mode == "coral" and lambda_domain > 0:
                    if target_iter is None and target_loader is not None:
                        target_iter = iter(target_loader)
                    if target_iter is not None:
                        try:
                            tgt_imgs, _, tgt_stage_lbl, _ = next(target_iter)
                        except StopIteration:
                            target_iter = iter(target_loader)
                            tgt_imgs, _, tgt_stage_lbl, _ = next(target_iter)
                        tgt_imgs = tgt_imgs.to(device, non_blocking=True)
                        with torch.no_grad():
                            tgt_feats = backbone(tgt_imgs)
                        tgt_domain_ids = torch.full(
                            (tgt_imgs.size(0),), encode_domain_ids("hv_clinical"), device=device, dtype=torch.long
                        )
                        _, tgt_aux = composer(
                            tgt_feats,
                            domain_ids=tgt_domain_ids,
                            stage_labels=list(tgt_stage_lbl),
                            align=False,
                        )
                        src_z = aux["z_g"]
                        tgt_z = tgt_aux["z_g"]
                        c_loss = coral_loss(src_z, tgt_z)
                        loss = loss + lambda_domain * c_loss
                        total_coral += float(c_loss.detach().cpu())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Validation
        backbone.eval()
        composer.eval()
        q_head.eval()
        if s_head:
            s_head.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for imgs, q, stage_lbl, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                q = q.to(device, non_blocking=True)
                feats = backbone(imgs)
                domain_ids = torch.full((imgs.size(0),), encode_domain_ids("hv_kaggle"), device=device, dtype=torch.long)
                fused, _ = composer(feats, domain_ids=domain_ids, stage_labels=list(stage_lbl), align=False)
                logits_q = q_head(fused)
                y_true.append(q.cpu())
                y_score.append(torch.sigmoid(logits_q).cpu())
        y_true_np = torch.cat(y_true).numpy()
        y_score_np = torch.cat(y_score).numpy()
        best_thresh, _ = find_best_threshold(y_true_np, y_score_np)
        val_metrics = compute_classification_metrics(y_true_np, y_score_np, threshold=best_thresh)
        if val_metrics["auprc"] > best_val:
            best_val = val_metrics["auprc"]
            best_state = {
                "epoch": epoch,
                "composer": composer.state_dict(),
                "q_head": q_head.state_dict(),
                "s_head": s_head.state_dict() if s_head else None,
                "cfg": cfg,
                "threshold": best_thresh,
            }
            torch.save(best_state, out_dir / "best_wp4.pth")
        log_msg = f"[epoch {epoch}] loss={total_loss/len(train_loader):.4f}"
        if domain_align_mode == "coral" and lambda_domain > 0 and len(train_loader) > 0:
            log_msg += f" coral_loss={total_coral/len(train_loader):.4f}"
        log_msg += f" val_auprc={val_metrics['auprc']:.4f}"
        print(log_msg)

    if best_state is None:
        raise RuntimeError("No best state saved.")
    composer.load_state_dict(best_state["composer"])
    q_head.load_state_dict(best_state["q_head"])
    if s_head and best_state["s_head"] is not None:
        s_head.load_state_dict(best_state["s_head"])
    threshold = best_state["threshold"]

    # Calibration on val
    scaler_temp = TemperatureScaler()
    logits_val = []
    labels_val = []
    with torch.no_grad():
        for imgs, q, stage_lbl, _ in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            feats = backbone(imgs)
            domain_ids = torch.full((imgs.size(0),), encode_domain_ids("hv_kaggle"), device=device, dtype=torch.long)
            fused, _ = composer(feats, domain_ids=domain_ids, stage_labels=list(stage_lbl), align=False)
            logits_q = q_head(fused)
            logits_val.append(logits_q.cpu())
            labels_val.append(q.cpu())
    logits_val_t = torch.cat(logits_val)
    labels_val_t = torch.cat(labels_val)
    temp_value = scaler_temp.fit(logits_val_t, labels_val_t)

    # Evaluate on Kaggle test and Clinical test
    y_true_k, y_score_k = run_eval(
        backbone, composer, q_head, s_head, test_loader, device, stage_classes, domain_str="hv_kaggle", temperature=temp_value
    )
    y_true_c, y_score_c = run_eval(
        backbone, composer, q_head, s_head, clinical_loader, device, stage_classes, domain_str="hv_clinical", temperature=temp_value
    )
    metrics_k = compute_classification_metrics(y_true_k, y_score_k, threshold=threshold)
    metrics_c = compute_classification_metrics(y_true_c, y_score_c, threshold=threshold)
    metrics_k["ece"] = compute_ece(y_score_k, y_true_k)
    metrics_c["ece"] = compute_ece(y_score_c, y_true_c)

    summary = {
        "threshold": threshold,
        "temperature": temp_value,
        "val_best_auprc": best_val,
        "kaggle_test": metrics_k,
        "clinical_test": metrics_c,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP4 multitask training with calibration.")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    train(cfg)


if __name__ == "__main__":
    main()
