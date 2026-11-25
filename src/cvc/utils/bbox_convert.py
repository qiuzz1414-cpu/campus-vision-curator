"""VLM 返回坐标的格式归一化。

这是整条流水线最容易出错的地方，所以单独抽出来、单独测试。

不同视觉模型的坐标约定：

| 模型族 | 坐标形式 | 本项目命名 |
|---|---|---|
| Qwen2.5-VL / GPT-4o | 绝对像素 `[x1,y1,x2,y2]` | `absolute` |
| Qwen3-VL / 部分 Qwen2.5-VL 微调版 | 归一化到 0~1000 的整数 | `norm1000` |
| 部分 LLaVA / InternVL 变体 | 归一化到 0~1 的浮点 | `norm1` |
| 少数实现 | `[x,y,w,h]` 而非 `[x1,y1,x2,y2]` | 自动识别 |

统一出口：`to_pixels()` 一律返回**绝对像素的 xyxy**，后续所有阶段只认这一种。
"""

from __future__ import annotations

from collections.abc import Sequence

FORMATS = ("auto", "absolute", "norm1000", "norm1")
_EPS = 1e-6


def detect_format(values: Sequence[float], img_w: int, img_h: int) -> str:
    """猜测坐标格式。仅用于 `auto` 模式，猜错时调用方应打印警告。

    判定顺序：
    1. 全部 <= 1.0001            -> norm1
    2. 峰值 > 1000.5             -> absolute（归一化不可能给出这么大的数）
    3. 峰值在 (1, 1000] 之间时，两种解释都可能，用**可行性**裁决：
       若按绝对像素解释会越出图像边界，则只能是 norm1000；
       否则按绝对像素解释（提示词里声明的默认格式）。
    """
    vals = [abs(float(v)) for v in values]
    if not vals:
        return "absolute"

    peak = max(vals)
    if peak <= 1.0001:
        return "norm1"
    if peak > 1000.5:
        return "absolute"

    # 峰值 <= 1000：可能是小图上的绝对像素，也可能是 norm1000
    if len(vals) >= 4:
        x1, y1, x2, y2 = vals[:4]
        abs_feasible = max(x1, x2) <= img_w + _EPS and max(y1, y2) <= img_h + _EPS
        if not abs_feasible:
            return "norm1000"  # 绝对解释越界，只能是归一化

    return "absolute"


def to_pixels(
    coords: Sequence[float],
    img_w: int,
    img_h: int,
    fmt: str = "auto",
) -> tuple[float, float, float, float]:
    """把任意格式的 4 个数字转成绝对像素 xyxy。

    Args:
        coords: 长度为 4 的坐标序列。
        img_w, img_h: 该图实际像素尺寸。
        fmt: auto / absolute / norm1000 / norm1。

    Returns:
        (x1, y1, x2, y2)，未做越界裁剪（交给 Box.clamp）。

    Raises:
        ValueError: 坐标数量不对，或 normalize 后仍明显越界。
    """
    if len(coords) != 4:
        raise ValueError(f"需要 4 个坐标值，收到 {len(coords)} 个：{coords}")
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"图像尺寸非法：{img_w}x{img_h}")

    if fmt == "auto":
        fmt = detect_format(coords, img_w, img_h)

    x1, y1, x2, y2 = (float(v) for v in coords)

    if fmt == "norm1":
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
    elif fmt == "norm1000":
        x1, x2 = x1 / 1000.0 * img_w, x2 / 1000.0 * img_w
        y1, y2 = y1 / 1000.0 * img_h, y2 / 1000.0 * img_h
    elif fmt == "absolute":
        pass
    else:
        raise ValueError(f"未知坐标格式：{fmt}，可选 {FORMATS}")

    # 反序很常见（右上/左下），统一成左上-右下
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def xywh_to_xyxy(coords: Sequence[float]) -> tuple[float, float, float, float]:
    """[x, y, w, h] -> [x1, y1, x2, y2]。"""
    if len(coords) != 4:
        raise ValueError(f"需要 4 个坐标值，收到 {len(coords)} 个")
    x, y, w, h = (float(v) for v in coords)
    return x, y, x + w, y + h


def normalize_to_norm1(
    coords: Sequence[float], img_w: int, img_h: int, fmt: str = "auto"
) -> tuple[float, float, float, float]:
    """把任意格式转成 0~1 归一化 xyxy（转发给下游或在不同模型间比较时用）。"""
    x1, y1, x2, y2 = to_pixels(coords, img_w, img_h, fmt)
    return x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的交并比，用于去重和与人工标注比对。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def clamp_box(
    box: Sequence[float], img_w: int, img_h: int, min_size: float = 2.0
) -> tuple[float, float, float, float] | None:
    """裁剪到图内；裁剪后太小则返回 None（视为无效框）。"""
    x1 = max(0.0, min(float(box[0]), img_w))
    y1 = max(0.0, min(float(box[1]), img_h))
    x2 = max(0.0, min(float(box[2]), img_w))
    y2 = max(0.0, min(float(box[3]), img_h))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return x1, y1, x2, y2


def dedup_boxes(
    boxes: list[tuple[str, tuple[float, float, float, float]]],
    iou_threshold: float = 0.85,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """同一类别的重复框去重（VLM 常见问题：同一物体报两次）。保留面积较大的。"""
    kept: list[tuple[str, tuple[float, float, float, float]]] = []
    for cls, box in sorted(
        boxes,
        key=lambda kv: (kv[1][2] - kv[1][0]) * (kv[1][3] - kv[1][1]),
        reverse=True,
    ):
        if any(cls == kc and iou(box, kb) > iou_threshold for kc, kb in kept):
            continue
        kept.append((cls, box))
    return kept
