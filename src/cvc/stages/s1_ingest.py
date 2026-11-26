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


def collect_files(
    paths: list[str],
    recursive: bool = True,
    exclude_root: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """把配置里的输入路径展开成文件列表，跳过隐藏文件和 workspace 内的产物。"""
    images: list[Path] = []
    videos: list[Path] = []
    skipped_by_rule = 0

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
            if _skip(f, exclude_root):
                skipped_by_rule += 1
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXT:
                images.append(f)
            elif ext in VIDEO_EXT:
                videos.append(f)

    if skipped_by_rule:
        log.info(f"按排除规则跳过 {skipped_by_rule} 个文件（位于 workspace/产物目录内或隐藏文件）")

    images.sort()
    videos.sort()
    return images, videos


PRODUCT_DIRS = {"workspace", "dataset", "labels", "overlays", "rejects", "runs", "review"}


def _skip(path: Path, exclude_root: Path | None = None) -> bool:
    """判断是否应跳过。

    只做两件事：
    1. 跳过隐藏文件
    2. 跳过**位于 workspace 内部**的文件（防止把自己的产物反向喂回来）

    早期版本用 `path.parts` 做集合判断，但 `parts` 包含绝对路径的每一层祖先。
    只要路径上任何一级目录叫 `images` / `dataset`（例如
    `/home/student/images/project/raw/`），所有文件都会被静默跳过，
    用户只看到"没有找到任何素材"，照片却在硬盘上。
    """
    if path.name.startswith("."):
        return True

    if exclude_root is not None:
        try:
            rel = path.resolve().relative_to(exclude_root.resolve())
        except (ValueError, OSError):
            return False  # 不在 workspace 内，保留
        # 只有 workspace 内的产物子目录才需要排除
        return bool(set(rel.parts[:-1]) & PRODUCT_DIRS) or rel.parts[:1] == ("images",)
    return False


def _ensure_heic_support() -> None:
    """HEIC 需要额外解码器。

    Pillow 自身不支持 HEIC，而 HEIC 是 iPhone 默认格式——即本模块 docstring
    声称的首要场景。没有解码器时给一条可操作的提示，而不是让用户看到
    "cannot identify image file" 这种莫名其妙的报错。
    """
    try:
        import pillow_heif  # noqa: F401

        pillow_heif.register_heif_opener()
    except ImportError:
        log.warning(
            "检测到 .heic 文件，但未安装解码器。iPhone 照片默认是 HEIC：\n"
            "          修法一：pip install pillow-heif\n"
            "          修法二：手机设置里改为「兼容性最佳」重新导出 JPEG"
        )


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


def _iter_frames(video: Path, fps: float):
    """按目标帧率**逐帧产出**视频帧，不在内存里堆整个视频。

    早期实现先把所有采样帧收集成 list 再返回：一分钟 1080p 视频按 1fps 抽帧
    就是 60 张 BGR 全帧（约 370MB），4K 长视频会直接把内存吃光。
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning(f"无法打开视频：{video}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(native_fps / fps))) if fps > 0 else 1
    log.info(f"{video.name}: 原生 {native_fps:.1f}fps，抽帧步长 {step}")

    idx = yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yielded += 1
                yield idx, frame
            idx += 1
    finally:
        cap.release()
    log.info(f"{video.name}: 实际抽取 {yielded} 帧")


def _write_jpeg(cv_img, dst: Path, quality: int) -> bool:
    """写 JPG 并**检查返回值**。

    cv2.imwrite 失败时只返回 False，不抛异常。最常见的失败原因是
    Windows 上路径含中文（OpenCV 用 ANSI 编码处理路径）。若不检查，
    下一行读取文件就会抛 FileNotFoundError，错误信息和真实原因完全脱节。
    """
    import cv2

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), cv_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        log.warning(
            f"写入失败：{dst}\n"
            "          常见原因是路径含中文/特殊字符（OpenCV 在 Windows 上不支持）。"
            "把 raw/ 和 workspace/ 换成英文路径即可。"
        )
    return bool(ok)


def run(cfg: Config, manifest: Manifest) -> dict:
    """执行接入阶段。已存在的记录会被跳过，因此可以反复运行。"""
    banner("阶段一 · 接入与标准化")
    cfg.ensure_dirs()

    images, videos = collect_files(
        cfg.input.paths, cfg.input.recursive, exclude_root=cfg.workspace_path
    )
    log.info(f"发现 {len(images)} 张图片、{len(videos)} 个视频")

    if any(p.suffix.lower() == ".heic" for p in images):
        _ensure_heic_support()

    if not images and not videos:
        paths_hint = ", ".join(cfg.input.paths)
        log.warning(
            f"没有找到任何素材。检查：\n"
            f"          1) 照片是否已放进 {paths_hint}\n"
            f"          2) 扩展名是否在支持范围内（{', '.join(sorted(IMAGE_EXT))}）\n"
            f"          3) 是否被排除规则跳过（隐藏文件、workspace 内部文件）"
        )
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
    # 逐帧处理并即时落盘 + 即时去重，避免整段视频的采样帧堆在内存里，
    # 也避免重跑时把同一批帧重复导入（早期版本漏了这一步）。
    import cv2

    for vid in videos:
        for frame_idx, frame in _iter_frames(vid, cfg.input.video_fps):
            if cfg.input.max_side and max(frame.shape[:2]) > cfg.input.max_side:
                scale = cfg.input.max_side / float(max(frame.shape[:2]))
                frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            rid = manifest.next_id()
            dst = cfg.images_dir / f"{rid}.jpg"
            if not _write_jpeg(frame, dst, cfg.input.jpeg_quality):
                skipped += 1
                continue

            digest = sha256_file(dst)
            if digest in seen_sha:
                # 同一段视频里相近的帧、或重复运行时的同一帧，都会落到这里
                log.debug(f"帧重复，跳过：{vid.name}#{frame_idx} == {seen_sha[digest]}")
                dst.unlink(missing_ok=True)
                skipped += 1
                continue

            h, w = frame.shape[:2]
            rec = Record(
                id=rid,
                image=str(dst.relative_to(cfg.workspace_path)),
                width=w,
                height=h,
                source=f"{vid.resolve()}#frame{frame_idx}",
                sha256=digest,
                status="ingested",
            )
            manifest.add(rec)
            seen_sha[digest] = rid
            added += 1

    manifest.save()
    log.info(f"新增 {added} 条，跳过 {skipped} 条，总计 {len(manifest)} 条")
    return {"added": added, "skipped": skipped}
