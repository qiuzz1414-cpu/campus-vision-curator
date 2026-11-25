"""VLM 调用与响应解析。

设计要点：
- 只依赖 OpenAI 兼容协议。本地 vLLM / 云端百炼 / 硅基流动 / OpenAI 都是同一套代码。
- 解析层要能扛住模型的"不听话"：```json 围栏、前后废话、单对象而非数组、字段名漂移。
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 图片编码


def image_to_data_url(path: str | Path, max_side: int = 1024) -> tuple[str, int, int]:
    """读图 → 可选缩小 → 转 base64 data URL。

    Returns:
        (data_url, sent_w, sent_h)。sent_* 是**实际发给模型**的尺寸，
        模型若返回绝对像素，必须用这个尺寸换算，而不是原图尺寸。
    """
    from io import BytesIO

    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max_side and max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=92)
        sent_w, sent_h = im.size

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", sent_w, sent_h


# ---------------------------------------------------------------- 提示词

_COORD_HINT = {
    "absolute": "Use integer pixel coordinates relative to the provided image.",
    "norm1000": (
        "Use integer coordinates normalized to the 0-1000 range "
        "(0 = left/top edge, 1000 = right/bottom edge)."
    ),
    "norm1": "Use float coordinates normalized to the 0-1 range.",
    "auto": "Use integer pixel coordinates relative to the provided image.",
}


def build_prompt(
    categories: list[str],
    distractors: list[str],
    bbox_format: str = "auto",
    extra: str = "",
) -> str:
    """构造标注提示词。

    刻意把"不要标什么"写进去——这是效果最明显的一句，
    因为 VLM 默认会热情地把画面里所有水果都标出来。
    """
    cats = ", ".join(f'"{c}"' for c in categories)
    hint = _COORD_HINT.get(bbox_format, _COORD_HINT["absolute"])

    parts = [
        "You are a dataset annotation assistant.",
        f"Detect every instance of these target categories: {cats}.",
    ]
    if distractors:
        dis = ", ".join(f'"{d}"' for d in distractors)
        parts.append(
            f"The following objects may also appear but MUST be ignored and MUST NOT be labeled: {dis}."
        )
    parts += [
        "Only output objects that clearly belong to the target categories.",
        "Return ONLY a single JSON object, no prose, no markdown code fence.",
        "Schema:",
        '{"objects": [{"label": "<target category>", "confidence": <0.0-1.0>, '
        '"bbox": [x1, y1, x2, y2]}]}',
        hint,
        'If no target object is present, return exactly {"objects": []}.',
    ]
    if extra.strip():
        parts.append(extra.strip())
    return "\n".join(parts)


# ---------------------------------------------------------------- 响应解析

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """从模型输出里挖出 JSON。容忍围栏、前后废话、尾部逗号。"""
    if not text or not text.strip():
        raise ValueError("模型返回为空")

    candidates: list[str] = []

    # 1) 优先取 markdown 围栏内的内容
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())

    # 2) 整体当 JSON
    candidates.append(text.strip())

    # 3) 从第一个 { / [ 到最后一个 } / ] 的切片
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i, j = text.find(open_ch), text.rfind(close_ch)
        if 0 <= i < j:
            candidates.append(text[i : j + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*([}\]])", r"\1", cand)  # 去尾部逗号
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"无法从模型输出中解析 JSON，前 200 字符：{text[:200]!r}")


@dataclass
class RawObject:
    """模型输出的原始对象（坐标还没换算）。"""

    label: str
    confidence: float
    bbox: list[float]


def normalize_objects(payload: Any) -> list[RawObject]:
    """把各种形态的 JSON 统一成 RawObject 列表。

    兼容形态：
    - {"objects": [...]}
    - {"detections": [...]} / {"results": [...]} / {"boxes": [...]}
    - 直接一个列表 [...]
    - 单个对象 {...}（模型只标到一个时会偷懒省略数组）
    """
    if isinstance(payload, dict):
        for key in ("objects", "detections", "results", "boxes", "annotations"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
        else:
            payload = [payload] if _looks_like_object(payload) else []

    if not isinstance(payload, list):
        raise ValueError(f"期望对象列表，收到 {type(payload).__name__}")

    out: list[RawObject] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        label = _pick(item, "label", "class", "category", "name", "cls")
        if not isinstance(label, str) or not label.strip():
            continue

        conf = _pick(item, "confidence", "score", "conf", "probability", default=1.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        if conf > 1.0:  # 百分制
            conf /= 100.0
        conf = max(0.0, min(1.0, conf))

        raw_box = _pick(item, "bbox", "box", "bbox_2d", "coordinates", "xyxy")
        box = _coerce_box(raw_box)
        if box is None:
            continue

        out.append(RawObject(label=label.strip(), confidence=conf, bbox=box))

    return out


def _pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _looks_like_object(d: dict) -> bool:
    return any(k in d for k in ("bbox", "box", "bbox_2d", "coordinates", "label", "class"))


def _coerce_box(v: Any) -> list[float] | None:
    """把 bbox 字段的各种写法拍平成 4 个数字。"""
    if v is None:
        return None
    if isinstance(v, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", v)
        v = nums
    if isinstance(v, dict):
        x1 = _pick(v, "x1", "xmin", "left", "x")
        y1 = _pick(v, "y1", "ymin", "top", "y")
        x2 = _pick(v, "x2", "xmax", "right")
        y2 = _pick(v, "y2", "ymax", "bottom")
        w = _pick(v, "w", "width")
        h = _pick(v, "h", "height")
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            v = [x1, y1, x2, y2]
        elif x1 is not None and y1 is not None and w is not None and h is not None:
            v = [x1, y1, x2 if x2 is not None else float(x1) + float(w),
                 y2 if y2 is not None else float(y1) + float(h)]
        else:
            return None
    if isinstance(v, (list, tuple)) and len(v) >= 4:
        try:
            return [float(x) for x in v[:4]]
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------- 服务端封装


def make_client(base_url: str, api_key: str, timeout: float = 120.0):
    """创建 OpenAI 兼容客户端。"""
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def list_models(base_url: str, api_key: str, timeout: float = 10.0) -> list[str]:
    """探测服务端可用模型名，用于 `cvc doctor` 自检。"""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return [m.id for m in client.models.list().data]
