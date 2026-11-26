"""阶段三 · VLM 自动标注。

异步并发调用 OpenAI 兼容接口，把每张图变成一组像素坐标的框。

三个关键细节（踩过才知道）：
1. **尺寸对齐**：发给模型的图会被缩小到 image_max_side。
   模型如果返回绝对像素，必须按**缩小后的尺寸**换算，否则框整体偏移。
2. **格式探测**：`auto` 模式下逐图猜坐标格式，猜成 norm1000 时打警告。
   生产环境建议在 recipe 里写死。
3. **低置信度标记**：不直接丢，而是打 `needs_review`，交给阶段四人工复核。
   自动标注的价值在于"省掉 80% 的点击"，不是"消灭人工"。
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from ..config import Config
from ..utils.bbox_convert import clamp_box, dedup_boxes, detect_format, to_pixels
from ..utils.logging import banner, get_logger
from ..utils.manifest import Box, Manifest
from ..utils.vlm import (
    build_prompt,
    extract_json,
    image_to_data_url,
    make_client,
    normalize_objects,
)

log = get_logger("label")


@dataclass
class LabelStats:
    requested: int = 0
    ok: int = 0
    failed: int = 0
    empty: int = 0
    boxes: int = 0
    low_conf: int = 0
    errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


async def _label_one(
    client,
    cfg: Config,
    rec,
    sem: asyncio.Semaphore,
    prompt: str,
    stats: LabelStats,
) -> None:
    """标注单张图。任何异常都只影响这一张，不中断整批。"""
    async with sem:
        img_path = cfg.workspace_path / rec.image
        try:
            data_url, sent_w, sent_h = image_to_data_url(
                img_path, cfg.label.image_max_side
            )
        except Exception as exc:
            log.warning(f"{rec.id} 读图失败：{exc}")
            rec.status = "labeled"
            rec.meta["label_error"] = f"read_error: {exc}"
            # 必须打 label_failed：否则 s5 会把这张读不出来的图当成"无目标的
            # 负样本"送进训练集，等于把坏数据洗成正常数据。
            rec.flag("label_failed")
            rec.flag("needs_review")
            stats.failed += 1
            return

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        kwargs: dict = {
            "model": cfg.label.model,
            "messages": messages,
            "temperature": cfg.label.temperature,
            "max_tokens": cfg.label.max_tokens,
        }
        if cfg.label.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(cfg.label.retries):
            try:
                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                payload = extract_json(text)
                objs = normalize_objects(payload)

                # 一个响应只判一次坐标格式。
                # 若逐个 object 独立猜，同一张图里可能一半按 absolute、
                # 一半按 norm1000 解析，框会出现整片偏移。
                fmt_used = cfg.label.bbox_format
                if fmt_used == "auto":
                    fmt_used = _detect_response_format(objs, sent_w, sent_h)

                sx = rec.width / sent_w
                sy = rec.height / sent_h

                boxes: list[tuple[str, tuple[float, float, float, float], float]] = []
                dropped_low = 0
                for obj in objs:
                    label = _match_category(obj.label, cfg.label.categories)
                    if label is None:
                        continue  # 模型标了不在目标里的类别，丢弃

                    # ★ min_conf 在这里真正生效
                    if obj.confidence < cfg.label.min_conf:
                        dropped_low += 1
                        continue

                    try:
                        px = to_pixels(obj.bbox, sent_w, sent_h, fmt_used)
                    except ValueError as exc:
                        log.debug(f"{rec.id} 坐标非法：{exc}")
                        continue

                    # 从「发给模型的尺寸」映射回「原图尺寸」
                    px = (px[0] * sx, px[1] * sy, px[2] * sx, px[3] * sy)

                    clipped = clamp_box(px, rec.width, rec.height)
                    if clipped is None:
                        continue
                    boxes.append((label, clipped, obj.confidence))

                boxes = dedup_boxes(boxes)

                rec.boxes = [
                    Box(
                        cls=c,
                        x1=b[0],
                        y1=b[1],
                        x2=b[2],
                        y2=b[3],
                        # ★ 保留模型给的置信度，不再写死 1.0
                        conf=conf,
                        source="auto",
                    )
                    for c, b, conf in boxes
                ]
                rec.status = "labeled"
                rec.meta["bbox_format_used"] = fmt_used
                rec.meta["label_model"] = cfg.label.model
                if dropped_low:
                    rec.meta["dropped_low_conf"] = dropped_low

                # 清掉上一次运行可能残留的标记，否则重新标注后旧状态会继续生效
                for stale in ("label_failed", "no_target", "needs_review", "mock_label"):
                    rec.unflag(stale)

                if not rec.boxes:
                    rec.flag("no_target")
                    stats.empty += 1
                # ★ review_conf 在这里真正生效
                if any(b.conf < cfg.label.review_conf for b in rec.boxes):
                    rec.flag("needs_review")
                    stats.low_conf += 1

                stats.ok += 1
                stats.boxes += len(rec.boxes)
                return

            except Exception as exc:  # 网络/解析/服务端错误都重试
                last_err = exc
                if attempt < cfg.label.retries - 1:
                    backoff = (2**attempt) + random.uniform(0, 0.5)
                    log.debug(f"{rec.id} 第 {attempt + 1} 次失败，{backoff:.1f}s 后重试：{exc}")
                    await asyncio.sleep(backoff)

        rec.status = "labeled"
        # retries 已由配置层保证 >= 1，但仍兜底：last_err 为 None 时写出
        # "None" 这种错误信息对学生毫无帮助
        rec.meta["label_error"] = (str(last_err) if last_err else "no attempt made")[:300]
        rec.flag("label_failed")
        rec.flag("needs_review")
        stats.failed += 1
        if len(stats.errors) < 5:
            stats.errors.append(f"{rec.id}: {last_err}")


def _detect_response_format(objs, sent_w: int, sent_h: int) -> str:
    """为一整个响应判定一次坐标格式。

    多目标取"多数票"：逐 object 判断一致时直接用；不一致时以出现次数最多的为准，
    并打警告——同一张图里混着两种坐标格式，本身就说明模型的输出不可靠。
    """
    votes: dict[str, int] = {}
    for obj in objs:
        fmt = detect_format(obj.bbox, sent_w, sent_h)
        votes[fmt] = votes.get(fmt, 0) + 1
    if not votes:
        return "absolute"
    if len(votes) > 1:
        log.warning(
            f"同一响应内坐标格式不一致：{votes}；按多数处理。"
            "建议在 recipe 里把 bbox_format 写死（auto 只用于首次探测）。"
        )
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _match_category(label: str, categories: list[str]) -> str | None:
    """把模型返回的类别名对齐到配置里的类名（忽略大小写、单复数、首尾空格）。"""
    norm = label.strip().lower().replace("_", " ").replace("-", " ")
    for cat in categories:
        c = cat.strip().lower().replace("_", " ").replace("-", " ")
        if norm == c or norm.rstrip("s") == c.rstrip("s"):
            return cat

    # 模型有时会返回 "a red apple" 这种短语。这里必须按**词边界**匹配：
    # 直接 `c in norm` 会让类别 "pea" 命中模型输出 "peach"，
    # 把完全无关的物体混进数据集。最长优先，避免 "apple" 抢先匹配 "apple pear"。
    tokens = set(norm.split())
    for cat in sorted(categories, key=len, reverse=True):
        c = cat.strip().lower().replace("_", " ").replace("-", " ")
        if " " in c:
            if f" {c} " in f" {norm} ":
                return cat
        elif c in tokens or c.rstrip("s") in tokens:
            return cat
    return None


def _mock_label(cfg: Config, rec, stats: LabelStats) -> None:
    """假标注：生成位置确定、尺寸合理的框，仅用于验证流水线连通性。

    不要用它训模型——框和图像内容毫无关系。它的唯一价值是：
    没有显卡、没有 API Key 的同学也能先把 s1→s6 的管道跑通，
    确认后面的导出/训练/报告环节没问题，再去申请算力做真标注。
    """
    import hashlib

    seed = int(hashlib.sha256(rec.id.encode()).hexdigest(), 16)
    cats = cfg.label.categories
    # 每张图 1~2 个框，类目轮换
    n = 1 + (seed % 2)
    rec.boxes = []
    for i in range(n):
        cls = cats[(seed + i) % len(cats)]
        # 在画面中部随机放一个占图宽 15%~35% 的框
        w = rec.width * (0.15 + ((seed >> (i * 3)) % 20) / 100.0)
        h = rec.height * (0.15 + ((seed >> (i * 5)) % 20) / 100.0)
        cx = rec.width * (0.3 + ((seed >> (i * 7)) % 40) / 100.0)
        cy = rec.height * (0.3 + ((seed >> (i * 9)) % 40) / 100.0)
        x1 = max(0.0, min(cx - w / 2, rec.width - w))
        y1 = max(0.0, min(cy - h / 2, rec.height - h))
        rec.boxes.append(
            Box(cls=cls, x1=x1, y1=y1, x2=x1 + w, y2=y1 + h, conf=0.9, source="auto")
        )

    rec.status = "labeled"
    rec.meta["label_model"] = "mock"
    rec.flag("mock_label")
    stats.ok += 1
    stats.boxes += len(rec.boxes)


def run(cfg: Config, manifest: Manifest, force: bool = False, limit: int | None = None) -> dict:
    """执行自动标注。默认只处理 status==cleaned 的记录。"""
    banner("阶段三 · VLM 自动标注")

    targets = [
        r for r in manifest.records
        if r.is_usable and (force or r.status == "cleaned")
    ]
    if limit:
        targets = targets[:limit]

    if not targets:
        log.info("没有待标注的记录（先跑 ingest + dedup）")
        return {"requested": 0}

    # ---- mock 后端：纯离线，只验证管道 ----
    if cfg.label.backend == "mock":
        log.warning("⚠️  使用 mock 后端生成的框与图像内容无关，仅用于验证流水线，切勿用于训练")
        stats = LabelStats(requested=len(targets))
        for rec in targets:
            _mock_label(cfg, rec, stats)
        manifest.save()
        log.info(f"假标注完成 {stats.ok} 张，{stats.boxes} 个框")
        return {"requested": stats.requested, "ok": stats.ok, "mock": True}

    log.info(
        f"待标注 {len(targets)} 张 | 模型 {cfg.label.model} | "
        f"并发 {cfg.label.concurrency} | 类别 {cfg.label.categories}"
    )

    prompt = build_prompt(
        cfg.label.categories,
        cfg.label.distractors,
        cfg.label.bbox_format,
        cfg.label.prompt_extra,
    )

    stats = LabelStats(requested=len(targets))

    async def _main() -> None:
        client = make_client(cfg.label.base_url, cfg.label.api_key, cfg.label.timeout)
        sem = asyncio.Semaphore(cfg.label.concurrency)
        try:
            from tqdm import tqdm

            tasks = [
                asyncio.create_task(_label_one(client, cfg, rec, sem, prompt, stats))
                for rec in targets
            ]
            with tqdm(total=len(tasks), desc="标注", unit="img") as bar:
                for coro in asyncio.as_completed(tasks):
                    await coro
                    bar.update(1)
        finally:
            await client.close()

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        log.warning("用户中断，已完成的标注会保存")

    manifest.save()

    log.info(
        f"成功 {stats.ok} | 失败 {stats.failed} | 无目标 {stats.empty} | "
        f"框总数 {stats.boxes} | 需复核 {stats.low_conf}"
    )
    for err in stats.errors:
        log.warning(f"样例错误：{err}")

    return {
        "requested": stats.requested,
        "ok": stats.ok,
        "failed": stats.failed,
        "empty": stats.empty,
        "boxes": stats.boxes,
        "needs_review": stats.low_conf,
    }
