"""阶段五 · 导出 YOLO 数据集。

把一个 manifest 变成 `ultralytics` 可以直接吃的目录：

    dataset/
    ├── data.yaml
    ├── stats.json          # 机器可读统计
    ├── stats.md            # 可直接贴进课程报告的表格
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/

三个容易忽略但很影响训练效果的点：
1. **每个类别都要在 train 里出现**。随机划分可能把某类全划到 val，训出来直接 0 召回。
2. **负样本（背景图）要按比例掺进去**，否则模型学会"总有东西可框"，误检率飙升。
3. **空标签文件是合法的**。负样本对应的 txt 就是空文件，不是缺失。
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict

import yaml

from ..config import Config
from ..utils.logging import banner, get_logger
from ..utils.manifest import Manifest

log = get_logger("export")


def _eligible(manifest: Manifest) -> list:
    """能进数据集的样本：未被剔除、已完成标注。"""
    return [
        r for r in manifest.records
        if r.is_usable and r.status in ("labeled", "exported")
        and "label_failed" not in r.flags
    ]


def stratified_split(
    positives: list, ratios: dict[str, float], seed: int
) -> dict[str, list]:
    """按主类别做分层划分，保证 train 覆盖所有类别。

    「主类别」= 该图框数最多的类。混合图（同时有苹果和柠檬）归到框多的一类。
    """
    rng = random.Random(seed)

    buckets: dict[str, list] = defaultdict(list)
    for rec in positives:
        counts: dict[str, int] = defaultdict(int)
        for b in rec.boxes:
            counts[b.cls] += 1
        dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "__none__"
        buckets[dominant].append(rec)

    out: dict[str, list] = {k: [] for k in ratios}

    for recs in buckets.values():
        rng.shuffle(recs)
        n = len(recs)

        # 小类别：至少保证 1 张进 train、1 张进 val（如果有 2 张以上）
        n_train = max(1, int(round(n * ratios.get("train", 0.8))))
        n_val = max(1, int(round(n * ratios.get("val", 0.15)))) if n >= 3 else 0
        n_train = min(n_train, n - n_val)

        out["train"].extend(recs[:n_train])
        cursor = n_train
        if "val" in out:
            out["val"].extend(recs[cursor : cursor + n_val])
            cursor += n_val
        if "test" in out:
            out["test"].extend(recs[cursor:])

    return out


def _sample_negatives(
    eligible: list, positives: set[str], ratios: dict[str, float], seed: int
) -> dict[str, list]:
    """从未标注到目标的图里抽负样本，掺进 train 和 val。"""
    rng = random.Random(seed + 1)
    pool = [r for r in eligible if r.id not in positives and not r.has_target]
    rng.shuffle(pool)

    n_train_pos = len(ratios["_train_pos"])
    n_neg = int(n_train_pos * ratios["_negative_ratio"])
    if n_neg <= 0 or not pool:
        return {"train": [], "val": []}

    picked = pool[:n_neg]
    n_val = max(1, int(n_neg * 0.2)) if len(picked) > 5 else 0
    return {"train": picked[: n_neg - n_val], "val": picked[n_neg - n_val :]}


def _write_sample(cfg: Config, rec, split: str, names: list[str], copy: bool) -> bool:
    """把一张图和它的标签写进 dataset。返回是否成功。"""
    src_img = cfg.workspace_path / rec.image
    if not src_img.exists():
        log.warning(f"{rec.id} 图片缺失，跳过：{src_img}")
        return False

    dst_img = cfg.dataset_dir / "images" / split / f"{rec.id}.jpg"
    dst_lbl = cfg.dataset_dir / "labels" / split / f"{rec.id}.txt"
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    if copy:
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
    else:
        try:
            if dst_img.is_symlink() or dst_img.exists():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())
        except OSError:
            shutil.copy2(src_img, dst_img)

    lines = []
    for b in rec.boxes:
        if b.cls not in names:
            log.debug(f"{rec.id} 出现未知类别 {b.cls}，已跳过")
            continue
        cid = names.index(b.cls)
        cx, cy, w, h = b.to_yolo(rec.width, rec.height)
        # YOLO 对越界值敏感，裁一下
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True


def _write_stats(cfg: Config, manifest: Manifest, splits: dict[str, list]) -> dict:
    names = cfg.class_names

    split_stats: dict[str, dict] = {}
    for split, recs in splits.items():
        per_class = {n: 0 for n in names}
        n_pos = n_neg = n_box = 0
        for rec in recs:
            if rec.has_target:
                n_pos += 1
            else:
                n_neg += 1
            for b in rec.boxes:
                if b.cls in per_class:
                    per_class[b.cls] += 1
                    n_box += 1
        split_stats[split] = {
            "images": len(recs),
            "positives": n_pos,
            "negatives": n_neg,
            "boxes": n_box,
            "boxes_per_class": per_class,
        }

    total_images = sum(v["images"] for v in split_stats.values())
    stats = {
        "project": cfg.project,
        "classes": names,
        "total_images": total_images,
        "splits": split_stats,
        "filtered_out": manifest.summary()["rejected"],
    }
    return stats


def _stats_markdown(stats: dict) -> str:
    names = stats["classes"]
    head = ["| 划分 | 图片数 | 正样本 | 负样本 | 框总数 | " + " | ".join(names) + " |"]
    head.append("|---|---|---|---|---|" + "---|" * len(names))

    rows = []
    for split, s in stats["splits"].items():
        cls_cells = " | ".join(str(s["boxes_per_class"].get(n, 0)) for n in names)
        rows.append(
            f"| {split} | {s['images']} | {s['positives']} | "
            f"{s['negatives']} | {s['boxes']} | {cls_cells} |"
        )

    tot_imgs = stats["total_images"]
    tot_box = sum(s["boxes"] for s in stats["splits"].values())
    tot_neg = sum(s["negatives"] for s in stats["splits"].values())
    cls_tot = " | ".join(
        str(sum(s["boxes_per_class"].get(n, 0) for s in stats["splits"].values()))
        for n in names
    )
    rows.append(f"| **合计** | **{tot_imgs}** | — | **{tot_neg}** | **{tot_box}** | {cls_tot} |")

    return "\n".join(
        [
            f"# 数据集统计 · {stats['project']}",
            "",
            f"- 类别：{', '.join(names)}",
            f"- 图片总数：{tot_imgs}",
            f"- 标注框总数：{tot_box}",
            f"- 负样本（背景图）：{tot_neg}",
            f"- 质量过滤剔除：{stats['filtered_out']} 张",
            "",
            *head,
            *rows,
            "",
            "> 本表由 `cvc export` 自动生成，可直接贴入实验报告。",
        ]
    )


def run(cfg: Config, manifest: Manifest, clean: bool = True) -> dict:
    """导出数据集。clean=True 时先清空旧的 dataset 目录。"""
    banner("阶段五 · 导出 YOLO 数据集")

    eligible = _eligible(manifest)
    positives = [r for r in eligible if r.has_target]

    if not positives:
        log.warning("没有任何带标注框的样本，无法导出（先跑 cvc label）")
        return {"images": 0}

    if clean and cfg.dataset_dir.exists():
        shutil.rmtree(cfg.dataset_dir)
        log.info(f"已清空旧数据集 {cfg.dataset_dir}")

    ratios = dict(cfg.export.split)
    positives_split = stratified_split(positives, ratios, cfg.export.seed)

    # 负样本按 train 正样本数×比例抽样
    extra = {"_train_pos": positives_split["train"], "_negative_ratio": cfg.export.negative_ratio}
    negatives = _sample_negatives(eligible, {r.id for r in positives}, extra, cfg.export.seed)

    splits: dict[str, list] = {}
    for split in ratios:
        splits[split] = list(positives_split.get(split, []))
        if split in negatives:
            splits[split].extend(negatives[split])

    names = cfg.class_names
    written = {k: 0 for k in splits}
    for split, recs in splits.items():
        for rec in recs:
            if _write_sample(cfg, rec, split, names, cfg.export.copy_images):
                written[split] += 1

    # ---- data.yaml ----
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "path": str(cfg.dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    if written.get("test"):
        data_yaml["test"] = "images/test"

    (cfg.dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # ---- 统计 ----
    stats = _write_stats(cfg, manifest, splits)
    (cfg.dataset_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.dataset_dir / "stats.md").write_text(_stats_markdown(stats), encoding="utf-8")

    for rec in eligible:
        if rec.id in {r.id for recs in splits.values() for r in recs}:
            rec.status = "exported"
    manifest.save()

    # ---- 日志 ----
    for split, n in written.items():
        s = stats["splits"][split]
        log.info(
            f"{split:<5} {n:>4} 张（正 {s['positives']} / 负 {s['negatives']}），"
            f"{s['boxes']} 个框"
        )

    log.info(f"data.yaml → {cfg.dataset_dir / 'data.yaml'}")
    log.info(f"统计表   → {cfg.dataset_dir / 'stats.md'}")

    # 类别覆盖检查：这是最容易踩的坑，主动报警
    train_classes = stats["splits"]["train"]["boxes_per_class"]
    missing = [n for n, c in train_classes.items() if c == 0]
    if missing:
        log.warning(f"⚠️ train 划分中缺少类别：{missing}，训练结果对应类别将无法识别")

    return stats
