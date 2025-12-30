"""
WP3 training entrypoint: frozen SSL backbone + WP3 modules + quality head.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from src.data.supervised_dataloader import _make_transforms
from src.models.frozen_backbone import build_frozen_backbone, FrozenBackboneEncoder
from src.models.wp3_modules import WP3FeatureComposer
from src.models.wp4_heads import QualityHead
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import set_seed


# -------------------------
# Helpers
# -------------------------

def resolve_path(image_path: str, domain: str, root_map: Dict[str, Path]) -> Optional[Path]:
    base = root_map.get(domain)
    if base is None:
        raise KeyError(f"domain '{domain}' not in root_map keys={list(root_map.keys())}")
    rel = Path(image_path)
    cand = base / rel
    if cand.exists():
        return cand
    # lowercase fallback
    lower_rel = Path(*[p.lower() for p in rel.parts])
    cand2 = base / lower_rel
    if cand2.exists():
        return cand2
    # insert alldata if missing after stage
    parts = list(lower_rel.parts)
    if parts and parts[0].startswith("ed") and (len(parts) < 2 or parts[1] != "alldata"):
        rel_with_alldata = Path(parts[0], "alldata", *parts[1:])
        cand3 = base / rel_with_alldata
        if cand3.exists():
            return cand3
    return None


def balanced_acc_threshold(y_true, y_score) -> Tuple[float, float]:
    import numpy as np
    from sklearn import metrics

    thresholds = np.unique(y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    best_th = 0.5
    best_bal_acc = -1.0
    for th in thresholds:
        y_pred = (y_score >= th).astype(int)
        bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_th = th
    return float(best_th), float(best_bal_acc)


class StageAwareDataset(Dataset):
    DOMAIN2ID = {"hv_kaggle": 0, "hv_clinical": 1}

    def __init__(
        self,
        csv_path: Path,
        split: str,
        domain: str,
        root_map: Dict[str, Path],
        label_col: str,
        transform,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = df[(df["domain"] == domain) & (df["split"] == split)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} domain={domain} in {csv_path}")
        paths: List[Optional[Path]] = []
        for ip in df["image_path"]:
            paths.append(resolve_path(ip, domain, root_map))
        df["abs_path"] = paths
        df = df[df["abs_path"].notna()].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No resolved samples for split={split} domain={domain} in {csv_path}")
        self.paths: List[Path] = df["abs_path"].tolist()
        self.labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(float).tolist()
        stage_col = "stage_raw" if "stage_raw" in df.columns else ("stage_group" if "stage_group" in df.columns else None)
        if stage_col:
            self.stages = df[stage_col].fillna("UNK").astype(str).tolist()
        else:
            self.stages = ["UNK"] * len(df)
        self.transform = transform
        if domain not in self.DOMAIN2ID:
            raise KeyError(f"domain '{domain}' not in DOMAIN2ID mapping {self.DOMAIN2ID}")
        self.domain_id = self.DOMAIN2ID[domain]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        from PIL import Image

        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label, self.domain_id, self.stages[idx]


# -------------------------
# Training
# -------------------------

def build_loaders(cfg: Dict):
    data_cfg = cfg["data"]
    img_size = data_cfg["img_size"]
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)
    root_map = {k: Path(v) for k, v in data_cfg["root_map"].items()}
    for k, v in root_map.items():
        if not v.exists():
            raise RuntimeError(f"[error] root_map['{k}'] does not exist: {v}")
    sup_csv = Path(data_cfg["supervised_csv"])
    cross_csv = Path(data_cfg["cross_csv"])
    label_col = data_cfg.get("label_col", "quality_label")

    train_tf = _make_transforms(img_size, train=True)
    eval_tf = _make_transforms(img_size, train=False)

    train_ds = StageAwareDataset(sup_csv, "train", "hv_kaggle", root_map, label_col, train_tf)
    val_ds = StageAwareDataset(sup_csv, "val", "hv_kaggle", root_map, label_col, eval_tf)
    test_ds = StageAwareDataset(sup_csv, "test", "hv_kaggle", root_map, label_col, eval_tf)
    clin_ds = StageAwareDataset(cross_csv, "test", "hv_clinical", root_map, label_col, eval_tf)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_ds, True),
        make_loader(val_ds, False),
        make_loader(test_ds, False),
        make_loader(clin_ds, False),
    )


def build_model(cfg: Dict, device: torch.device) -> Tuple[FrozenBackboneEncoder, WP3FeatureComposer, nn.Module]:
    mcfg = cfg["model"]
    wp3cfg = cfg["wp3"]
    backbone = build_frozen_backbone(
        name=mcfg.get("backbone_name", "vit_base_patch16_224"),
        pretrained=False if mcfg.get("mode", "ssl") == "ssl" else True,
        checkpoint=Path(mcfg["ssl_checkpoint"]) if mcfg.get("mode", "ssl") == "ssl" else None,
        from_byol_ckpt=mcfg.get("from_byol_ckpt", True),
    ).to(device)
    composer = WP3FeatureComposer(
        in_dim=backbone.feature_dim,
        token_dim=wp3cfg.get("token_dim", backbone.feature_dim),
        adapter_cfg=wp3cfg.get("adapter", {}),
        morph_cfg=wp3cfg.get("morph", {}),
        gen_cfg=wp3cfg.get("gen", {}),
    ).to(device)
    head = QualityHead(in_dim=composer.out_dim).to(device)
    print(
        f"[model] backbone_dim={backbone.feature_dim} token_dim={wp3cfg.get('token_dim', backbone.feature_dim)} fused_dim={composer.out_dim}"
    )
    return backbone, composer, head


def run_epoch(loader, backbone, composer, head, optimizer, criterion, device, scaler=None, train=True, enable_align=False, lambda_domain=0.0):
    if train:
        head.train()
        composer.train()
    else:
        head.eval()
        composer.eval()
    backbone.eval()
    total_loss = 0.0
    y_true = []
    y_score = []
    domain_loss = torch.tensor(0.0, device=device)
    for imgs, labels, domain_ids, stages in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        domain_ids = domain_ids.to(device, non_blocking=True).long().view(-1)
        stages_list = list(stages)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            feats = backbone(imgs)
            fused, aux = composer(feats, domain_ids=domain_ids, stage_labels=stages_list, align=enable_align)
            assert fused.dim() == 2, f"fused should be 2D, got {fused.shape}"
            assert fused.size(1) == composer.out_dim, f"fused dim {fused.size(1)} != composer.out_dim {composer.out_dim}"
            logits = head(fused).view(-1, 1)
            loss = criterion(logits, labels)
            if enable_align and "domain_logits" in aux and lambda_domain > 0:
                dom_logits = aux["domain_logits"]
                dom_targets = domain_ids
                d_loss = nn.functional.cross_entropy(dom_logits, dom_targets)
                loss = loss + lambda_domain * d_loss
                domain_loss = domain_loss + d_loss.detach()
        if train:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        y_true.append(labels.detach().cpu().view(-1))
        y_score.append(torch.sigmoid(logits.detach()).cpu().view(-1))
    y_true_np = torch.cat(y_true).numpy()
    y_score_np = torch.cat(y_score).numpy()
    avg_loss = total_loss / len(y_true_np)
    return avg_loss, y_true_np, y_score_np, float(domain_loss)


def main():
    parser = argparse.ArgumentParser(description="WP3 training with frozen SSL backbone.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Runtime summary
    print(f"[startup] root_map domains={list(cfg['data']['root_map'].keys())}")
    print(f"[startup] DOMAIN2ID mapping={StageAwareDataset.DOMAIN2ID}")
    enable_align = cfg["train"].get("enable_domain_align", False) and cfg["wp3"]["gen"].get("use_domain_align", True)
    if not enable_align:
        print("[startup] domain alignment DISABLED for this run; lambda_domain will be ignored.")
    train_loader, val_loader, test_loader, clin_loader = build_loaders(cfg)
    train_loader, val_loader, test_loader, clin_loader = build_loaders(cfg)

    backbone, composer, head = build_model(cfg, device)
    if cfg["model"].get("freeze_backbone", True):
        for p in backbone.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(
        list(composer.parameters()) + list(head.parameters()),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    use_amp = cfg["train"].get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_weight = cfg["train"].get("pos_weight", None)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    )

    out_dir = Path(cfg["logging"]["out_dir"]) / cfg["logging"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    best_bal_acc = -1.0
    best_state = None
    enable_align = enable_align
    lambda_domain = cfg["train"].get("lambda_domain", 0.0)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, _, _, dom_loss = run_epoch(
            train_loader, backbone, composer, head, optimizer, criterion, device, scaler=scaler, train=True, enable_align=enable_align, lambda_domain=lambda_domain
        )
        val_loss, y_true_val, y_score_val, _ = run_epoch(
            val_loader, backbone, composer, head, optimizer, criterion, device, scaler=None, train=False, enable_align=False
        )
        th, bal_acc = balanced_acc_threshold(y_true_val, y_score_val)
        val_metrics = compute_classification_metrics(y_true_val, y_score_val, threshold=th)
        val_metrics["bal_acc"] = bal_acc
        val_pos_rate = float(y_true_val.mean()) if len(y_true_val) else 0.0
        print(
            "[epoch {e}] train_loss={tl:.4f} val_loss={vl:.4f} "
            "val_auroc={auroc:.4f} val_auprc={auprc:.4f} val_f1={f1:.4f} "
            "val_acc={acc:.4f} val_bal_acc={bal:.4f} th={th:.3f} val_pos_rate={pr:.3f} dom_loss={dl:.4f}".format(
                e=epoch,
                tl=train_loss,
                vl=val_loss,
                auroc=val_metrics["auroc"],
                auprc=val_metrics["auprc"],
                f1=val_metrics["f1"],
                acc=val_metrics["acc"],
                bal=bal_acc,
                th=th,
                pr=val_pos_rate,
                dl=dom_loss,
            )
        )
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = {
                "epoch": epoch,
                "composer": composer.state_dict(),
                "head": head.state_dict(),
                "threshold": th,
                "val_metrics": val_metrics,
                "cfg": cfg,
            }
            torch.save(best_state, out_dir / "best_wp3.pth")

    if best_state is None:
        raise RuntimeError("No best state recorded.")
    composer.load_state_dict(best_state["composer"])
    head.load_state_dict(best_state["head"])
    threshold = best_state["threshold"]

    # Eval in-domain test and cross-domain
    _, y_true_test, y_score_test, _ = run_epoch(
        test_loader, backbone, composer, head, optimizer, criterion, device, scaler=None, train=False, enable_align=False
    )
    _, y_true_clin, y_score_clin, _ = run_epoch(
        clin_loader, backbone, composer, head, optimizer, criterion, device, scaler=None, train=False, enable_align=False
    )
    test_metrics = compute_classification_metrics(y_true_test, y_score_test, threshold=threshold)
    clin_metrics = compute_classification_metrics(y_true_clin, y_score_clin, threshold=threshold)

    summary = {
        "config": cfg,
        "threshold": threshold,
        "val_metrics": best_state["val_metrics"],
        "test_kaggle": test_metrics,
        "test_clinical": clin_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
