"""阶段一 · 接入与标准化。

把形态各异的原始素材统一成「一张规整的 JPG + 一条 manifest 记录」：
- 手机照片：修 EXIF 旋转（不修的话 90% 的 iPhone 竖拍图是横的，标出来的框全错）
- 视频：按固定帧率抽帧
- 统一转 RGB、限制最长边、重编码为 JPG
- 用 sha256 干掉完全重复的文件
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..utils.logging import banner, get_logger
from ..utils.manifest import Manifest, Record, sha256_file

log = get_logger("ingest")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def collect_files(paths: list[str], recursive: bool = True) -> tuple[list[Path], list[Path]]:
    """把配置里的输入路径展开成文件列表，跳过隐藏文件和已生成的产物。"""
    images: list[Path] = []
    videos: list[Path] = []

    for raw in paths:
        p = Path(raw).expanduser()
        if not p.exists():
            log.warning(f"输入路径不存在，跳过：{p}")
            continue
        if p.is_file():
            candidates = [p]
        else:
            pattern = "**/*" if recursive else "*"
            candidates = [f for f in p.glob(pattern) if f.is_file()]

        for f in candidates:
            if _skip(f):
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXT:
                images.append(f)
            elif ext in VIDEO_EXT:
                videos.append(f)

    images.sort()
    videos.sort()
    return images, videos


def _skip(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    parts = set(path.parts)
    # 避免把 workspace / 导出数据集反向喂回来，造成无限自我吞噬
    return bool(parts & {"workspace", "dataset", "images", "labels", "overlays", "rejects", "runs"})


def _save_normalized(src: Path, dst: Path, max_side: int, quality: int) -> tuple[int, int]:
    """读图 → 修 EXIF → 转 RGB → 限边 → 存 JPG。返回最终宽高。"""
    from PIL import Image, ImageOps

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)  # 关键：修正手机照片方向
        im = im.convert("RGB")
        w, h = im.size
        if max_side and max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=quality, optimize=True)
        return im.size


def _extract_frames(video: Path, fps: float) -> list[tuple[int, any]]:
    """按目标帧率抽取视频帧。返回 [(帧号, BGR ndarray)]。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning(f"无法打开视频：{video}")
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / fps))) if fps > 0 else 1

    frames: list[tuple[int, any]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()

    log.info(f"{video.name}: 原生 {native_fps:.1f}fps，抽帧步长 {step}，共 {len(frames)} 帧")
    return frames


def run(cfg: Config, manifest: Manifest) -> dict:
    """执行接入阶段。已存在的记录会被跳过，因此可以反复运行。"""
    banner("阶段一 · 接入与标准化")
    cfg.ensure_dirs()

    images, videos = collect_files(cfg.input.paths, cfg.input.recursive)
    log.info(f"发现 {len(images)} 张图片、{len(videos)} 个视频")

    if not images and not videos:
        log.warning("没有找到任何素材。请把照片放进 recipe 中 input.paths 指定的目录。")
        return {"added": 0, "skipped": 0}

    seen_sha: dict[str, str] = {
        r.sha256: r.id for r in manifest.records if r.sha256
    }
    added = skipped = 0

    # ---- 图片 ----
    for src in images:
        digest = sha256_file(src)
        if digest in seen_sha:
            log.debug(f"完全重复，跳过：{src.name} == {seen_sha[digest]}")
            skipped += 1
            continue

        rid = manifest.next_id()
        dst = cfg.images_dir / f"{rid}.jpg"
        try:
            w, h = _save_normalized(src, dst, cfg.input.max_side, cfg.input.jpeg_quality)
        except Exception as exc:  # 坏图不该让整批任务崩掉
            log.warning(f"读取失败，跳过 {src.name}：{exc}")
            skipped += 1
            continue

        rec = Record(
            id=rid,
            image=str(dst.relative_to(cfg.workspace_path)),
            width=w,
            height=h,
            source=str(src.resolve()),
            sha256=digest,
            status="ingested",
        )
        manifest.add(rec)
        seen_sha[digest] = rid
        added += 1

    # ---- 视频抽帧 ----
    import cv2

    for vid in videos:
        for frame_idx, frame in _extract_frames(vid, cfg.input.video_fps):
            rid = manifest.next_id()
            dst = cfg.images_dir / f"{rid}.jpg"

            if cfg.input.max_side and max(frame.shape[:2]) > cfg.input.max_side:
                scale = cfg.input.max_side / float(max(frame.shape[:2]))
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.input.jpeg_quality])
            h, w = frame.shape[:2]

            rec = Record(
                id=rid,
                image=str(dst.relative_to(cfg.workspace_path)),
                width=w,
                height=h,
                source=f"{vid.resolve()}#frame{frame_idx}",
                sha256=sha256_file(dst),
                status="ingested",
            )
            manifest.add(rec)
            added += 1

    manifest.save()
    log.info(f"新增 {added} 条，跳过 {skipped} 条，总计 {len(manifest)} 条")
    return {"added": added, "skipped": skipped}
