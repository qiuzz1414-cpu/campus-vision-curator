"""阶段六 · 训练与指标汇总。

封装 Ultralytics YOLO 训练，并自动把 `results.csv` 转成一张可贴进报告的 Markdown 表。

依赖是惰性的：只有真正执行这一阶段才会 import ultralytics，
所以没装训练依赖的同学依然可以跑完 s1~s5。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..config import Config
from ..utils.logging import banner, get_logger

log = get_logger("train")


def _resolve_device(device: str) -> str | None:
    """auto -> 有 GPU 用 0，否则 cpu。返回 None 表示交给 ultralytics 自己判断。"""
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return None


def _summarize_results(run_dir: Path) -> dict:
    """从 results.csv 提取关键指标。"""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({(k or "").strip(): v for k, v in row.items()})

    if not rows:
        return {}

    def _num(row: dict, *keys: str) -> float | None:
        for k in keys:
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except (TypeError, ValueError):
                    continue
        return None

    last = rows[-1]
    best_map = max(
        (_num(r, "metrics/mAP50-95(B)", "mAP50-95") or 0.0 for r in rows),
        default=0.0,
    )

    return {
        "epochs_run": len(rows),
        "best_mAP50_95": round(best_map, 4),
        "final": {
            "precision": _num(last, "metrics/precision(B)", "precision"),
            "recall": _num(last, "metrics/recall(B)", "recall"),
            "mAP50": _num(last, "metrics/mAP50(B)", "mAP50"),
            "mAP50_95": _num(last, "metrics/mAP50-95(B)", "mAP50-95"),
        },
        "paths": {
            "best_weights": str(run_dir / "weights" / "best.pt"),
            "last_weights": str(run_dir / "weights" / "last.pt"),
            "results_csv": str(csv_path),
        },
    }


def _markdown(summary: dict, cfg: Config) -> str:
    f = summary.get("final", {})
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    return "\n".join(
        [
            f"# 训练结果 · {cfg.project}",
            "",
            f"- 基础模型：`{cfg.train.model}`",
            f"- epochs：{cfg.train.epochs}（实际完成 {summary.get('epochs_run', '?')}）",
            f"- imgsz：{cfg.train.imgsz} · batch：{cfg.train.batch}",
            "",
            "| 指标 | 数值 |",
            "|---|---|",
            f"| mAP@50 | {fmt(f.get('mAP50'))} |",
            f"| mAP@50-95 | {fmt(f.get('mAP50_95'))} |",
            f"| Precision | {fmt(f.get('precision'))} |",
            f"| Recall | {fmt(f.get('recall'))} |",
            f"| 最佳 mAP@50-95 | {summary.get('best_mAP50_95', 0):.4f} |",
            "",
            f"- 最佳权重：`{summary.get('paths', {}).get('best_weights', '—')}`",
            "",
            "> 由 `cvc train` 自动生成。",
        ]
    )


def run(cfg: Config, train_overrides: dict | None = None) -> dict:
    """执行训练。"""
    banner("阶段六 · 训练 YOLO")

    if not cfg.train.enabled:
        log.info("recipe 中 train.enabled=false，跳过训练")
        return {"skipped": True}

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("未安装 ultralytics。请执行：pip install -e '.[train]'")
        return {"skipped": True, "reason": "missing_ultralytics"}

    data_yaml = cfg.dataset_dir / "data.yaml"
    if not data_yaml.exists():
        log.error(f"找不到 {data_yaml}（先执行 cvc export）")
        return {"skipped": True, "reason": "missing_dataset"}

    device = _resolve_device(cfg.train.device)
    log.info(
        f"模型 {cfg.train.model} | epochs {cfg.train.epochs} | "
        f"imgsz {cfg.train.imgsz} | batch {cfg.train.batch} | device {device}"
    )

    kwargs = {
        "data": str(data_yaml),
        "epochs": cfg.train.epochs,
        "imgsz": cfg.train.imgsz,
        "batch": cfg.train.batch,
        "patience": cfg.train.patience,
        "workers": cfg.train.workers,
        "project": str(cfg.workspace_path / "runs"),
        "name": "train",
        "exist_ok": True,
        "plots": True,
    }
    if device:
        kwargs["device"] = device
    if train_overrides:
        kwargs.update(train_overrides)

    model = YOLO(cfg.train.model)
    results = model.train(**kwargs)

    # 定位本次 run 目录
    run_dir = Path(getattr(results, "save_dir", cfg.workspace_path / "runs" / "train"))
    summary = _summarize_results(run_dir)

    if summary:
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "summary.md").write_text(_markdown(summary, cfg), encoding="utf-8")

        f = summary["final"]
        log.info(
            "训练完成 | "
            f"mAP50={f.get('mAP50') or 0:.4f} "
            f"mAP50-95={f.get('mAP50_95') or 0:.4f} "
            f"P={f.get('precision') or 0:.4f} R={f.get('recall') or 0:.4f}"
        )
        log.info(f"最佳权重 → {summary['paths']['best_weights']}")
        log.info(f"指标表   → {run_dir / 'summary.md'}")
    else:
        log.warning("训练结束但未找到 results.csv，无法汇总指标")

    return summary or {"run_dir": str(run_dir)}
