"""阶段二 · 质量过滤与去重。

三道闸门，从便宜到贵：
1. **尺寸**：太小的图（缩略图、聊天截图）直接弃用
2. **清晰度**：Laplacian 方差低于阈值 → 糊了
3. **重复**：感知哈希（pHash）汉明距离近 → 连拍/同一场景反复拍

被剔除的图不删除，移到 `rejects/` 并在 manifest 里记 `reject_reason`。
保留证据，报告里要用"过滤掉多少张、为什么"。
"""

from __future__ import annotations

import shutil

import numpy as np

from ..config import Config
from ..utils.logging import banner, get_logger
from ..utils.manifest import Manifest

log = get_logger("dedup")


def laplacian_variance(gray: np.ndarray) -> float:
    """清晰度指标：拉普拉斯响应的方差。越大越清晰。

    经验阈值：清晰照片 > 100，轻微虚焦 30~100，明显糊 < 30。
    """
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _load_gray(cfg: Config, rec) -> np.ndarray | None:
    import cv2

    path = cfg.workspace_path / rec.image
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning(f"无法读取：{path}")
    return img


def _phash(cfg: Config, rec) -> str | None:
    import imagehash
    from PIL import Image

    try:
        with Image.open(cfg.workspace_path / rec.image) as im:
            return str(imagehash.phash(im.convert("RGB"), hash_size=8))
    except Exception as exc:
        log.debug(f"pHash 计算失败 {rec.id}：{exc}")
        return None


def _clip_embeddings(cfg: Config, records: list) -> dict[str, np.ndarray]:
    """可选的 CLIP 语义去重。需要 `pip install -e '.[clip]'`。"""
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        log.warning("未安装 open_clip_torch，跳过 CLIP 去重（如需：pip install -e '.[clip]'）")
        return {}

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.filter.clip.model, pretrained=cfg.filter.clip.pretrained
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embs: dict[str, np.ndarray] = {}
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            tensors = []
            for rec in chunk:
                with Image.open(cfg.workspace_path / rec.image) as im:
                    tensors.append(preprocess(im.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            for rec, vec in zip(chunk, feats.cpu().numpy(), strict=False):
                embs[rec.id] = vec
    return embs


def _reject(cfg: Config, rec, reason: str) -> None:
    rec.reject_reason = reason
    src = cfg.workspace_path / rec.image
    if src.exists():
        dst = cfg.rejects_dir / src.name
        try:
            shutil.move(str(src), str(dst))
            rec.meta["rejected_from"] = rec.image
            rec.image = str(dst.relative_to(cfg.workspace_path))
        except OSError as exc:
            log.debug(f"移动被拒图失败 {src}：{exc}")


def run(cfg: Config, manifest: Manifest, force: bool = False) -> dict:
    """执行过滤与去重。force=True 时重新计算全部（默认只处理未清洗过的）。"""
    banner("阶段二 · 质量过滤与去重")
    cfg.ensure_dirs()

    targets = [
        r for r in manifest.records
        if r.reject_reason is None and (force or r.status == "ingested")
    ]
    if not targets:
        log.info("没有待处理的记录")
        return {"processed": 0}

    log.info(f"待处理 {len(targets)} 条")

    # ---- 闸门 1：清晰度 + 尺寸 ----
    n_blur = n_small = 0
    for rec in targets:
        if min(rec.width, rec.height) < cfg.filter.min_side:
            _reject(cfg, rec, f"too_small_{rec.width}x{rec.height}")
            n_small += 1
            continue

        gray = _load_gray(cfg, rec)
        if gray is None:
            _reject(cfg, rec, "unreadable")
            continue

        score = laplacian_variance(gray)
        rec.blur = round(score, 2)
        if score < cfg.filter.blur_threshold:
            _reject(cfg, rec, f"blurry_{score:.1f}")
            n_blur += 1

    log.info(f"尺寸过小 {n_small} 条，模糊 {n_blur} 条")

    # ---- 闸门 2：pHash 去重 ----
    survivors = [r for r in targets if r.reject_reason is None]
    n_dup = 0
    kept_hashes: list[tuple[str, str]] = []  # (rec_id, hash_str)

    import imagehash

    for rec in survivors:
        h = _phash(cfg, rec)
        if h is None:
            continue
        rec.phash = h

        # 线性比对在万级数据量下够用；百万级再换 BK-tree
        dup_of = None
        for kept_id, kept_h in kept_hashes:
            dist = imagehash.hex_to_hash(h) - imagehash.hex_to_hash(kept_h)
            if dist <= cfg.filter.phash_distance:
                dup_of = kept_id
                break

        if dup_of:
            _reject(cfg, rec, f"duplicate_of_{dup_of}")
            n_dup += 1
        else:
            kept_hashes.append((rec.id, h))

    log.info(f"感知哈希重复 {n_dup} 条")

    # ---- 闸门 3：CLIP 语义去重（可选）----
    n_clip = 0
    if cfg.filter.clip.enabled:
        pool = [r for r in survivors if r.reject_reason is None]
        embs = _clip_embeddings(cfg, pool)
        seen: list[tuple[str, np.ndarray]] = []
        for rec in pool:
            vec = embs.get(rec.id)
            if vec is None:
                continue
            dup_of = None
            for kept_id, kept_vec in seen:
                if float(np.dot(vec, kept_vec)) >= cfg.filter.clip.threshold:
                    dup_of = kept_id
                    break
            if dup_of:
                _reject(cfg, rec, f"clip_dup_of_{dup_of}")
                n_clip += 1
            else:
                seen.append((rec.id, vec))
        log.info(f"CLIP 语义重复 {n_clip} 条")

    # ---- 收尾 ----
    for rec in targets:
        if rec.reject_reason is None:
            rec.status = "cleaned"
            rec.unflag("blurry")

    manifest.save()
    passed = sum(1 for r in targets if r.reject_reason is None)
    log.info(f"通过 {passed} / {len(targets)} 条")
    return {
        "processed": len(targets),
        "passed": passed,
        "blurry": n_blur,
        "too_small": n_small,
        "duplicate": n_dup,
        "clip_duplicate": n_clip,
    }
