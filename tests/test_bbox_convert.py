"""坐标转换单元测试——这是整条流水线最容易出错的地方。"""

from __future__ import annotations

import pytest

from cvc.utils.bbox_convert import (
    clamp_box,
    dedup_boxes,
    detect_format,
    iou,
    normalize_to_norm1,
    to_pixels,
    xywh_to_xyxy,
)


class TestDetectFormat:
    def test_norm1(self):
        assert detect_format([0.1, 0.2, 0.5, 0.8], 1280, 720) == "norm1"

    def test_boundary_one(self):
        # 恰好等于 1.0 应判为 norm1
        assert detect_format([0.0, 0.0, 1.0, 1.0], 640, 480) == "norm1"

    def test_norm1000(self):
        assert detect_format([100, 200, 900, 800], 1280, 720) == "norm1000"

    def test_absolute_when_exceeds_image(self):
        # 值超过图像尺寸，只能是绝对像素
        assert detect_format([100, 200, 1300, 800], 1280, 720) == "absolute"

    def test_absolute_large(self):
        assert detect_format([1200, 300, 1900, 1000], 2000, 1500) == "absolute"


class TestToPixels:
    def test_norm1_scaling(self):
        out = to_pixels([0.1, 0.2, 0.5, 0.6], 1000, 500, "norm1")
        assert out == pytest.approx((100.0, 100.0, 500.0, 300.0))

    def test_norm1000_scaling(self):
        out = to_pixels([100, 200, 500, 600], 1000, 500, "norm1000")
        assert out == pytest.approx((100.0, 100.0, 500.0, 300.0))

    def test_absolute_identity(self):
        out = to_pixels([10, 20, 300, 400], 1000, 500, "absolute")
        assert out == pytest.approx((10.0, 20.0, 300.0, 400.0))

    def test_reversed_coords_are_fixed(self):
        # 模型偶尔返回右下-左上的顺序
        out = to_pixels([500, 300, 100, 100], 1000, 500, "norm1")
        assert out[0] < out[2] and out[1] < out[3]

    def test_auto_mode(self):
        out = to_pixels([0.25, 0.25, 0.75, 0.75], 400, 400, "auto")
        assert out == pytest.approx((100.0, 100.0, 300.0, 300.0))

    def test_wrong_arity_raises(self):
        with pytest.raises(ValueError):
            to_pixels([1, 2, 3], 100, 100)

    def test_bad_dims_raises(self):
        with pytest.raises(ValueError):
            to_pixels([1, 2, 3, 4], 0, 100)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            to_pixels([1, 2, 3, 4], 100, 100, "normalized")


class TestXYWH:
    def test_convert(self):
        assert xywh_to_xyxy([10, 20, 30, 40]) == (10.0, 20.0, 40.0, 60.0)


class TestNormalize:
    def test_roundtrip(self):
        norm = normalize_to_norm1([100, 100, 300, 300], 400, 400, "absolute")
        assert norm == pytest.approx((0.25, 0.25, 0.75, 0.75))


class TestIoU:
    def test_identical(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint(self):
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_half_overlap(self):
        # 两个 10x10 框重叠 5x10 = 50，并集 150
        assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


class TestClamp:
    def test_clips_negative(self):
        assert clamp_box([-10, -10, 50, 50], 100, 100) == (0.0, 0.0, 50.0, 50.0)

    def test_clips_overflow(self):
        assert clamp_box([50, 50, 200, 200], 100, 100) == (50.0, 50.0, 100.0, 100.0)

    def test_too_small_returns_none(self):
        assert clamp_box([10, 10, 11, 11], 100, 100, min_size=2.0) is None

    def test_fully_outside_returns_none(self):
        assert clamp_box([200, 200, 300, 300], 100, 100) is None


class TestDedup:
    def test_removes_near_duplicates(self):
        boxes = [
            ("apple", (0.0, 0.0, 10.0, 10.0)),
            ("apple", (0.5, 0.5, 10.5, 10.5)),  # 几乎同一框
            ("apple", (50.0, 50.0, 60.0, 60.0)),  # 另一个苹果
        ]
        kept = dedup_boxes(boxes, iou_threshold=0.8)
        assert len(kept) == 2

    def test_keeps_different_classes(self):
        boxes = [
            ("apple", (0.0, 0.0, 10.0, 10.0)),
            ("lemon", (0.0, 0.0, 10.0, 10.0)),
        ]
        assert len(dedup_boxes(boxes, iou_threshold=0.8)) == 2
