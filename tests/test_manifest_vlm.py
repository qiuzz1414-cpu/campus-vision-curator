"""Manifest 数据契约与 VLM 响应解析的测试。"""

from __future__ import annotations

import json

import pytest

from cvc.utils.manifest import Box, Manifest, Record
from cvc.utils.vlm import extract_json, normalize_objects


class TestBox:
    def test_yolo_roundtrip(self):
        b = Box(cls="apple", x1=100, y1=50, x2=300, y2=250)
        cx, cy, w, h = b.to_yolo(400, 500)
        assert (cx, cy, w, h) == pytest.approx((0.5, 0.3, 0.5, 0.4))

        line = f"0 {cx} {cy} {w} {h}"
        back = Box.from_yolo(line, 400, 500, ["apple", "lemon"])
        assert back.cls == "apple"
        assert back.x1 == pytest.approx(100)
        assert back.y2 == pytest.approx(250)

    def test_area_and_dims(self):
        b = Box(cls="lemon", x1=10, y1=10, x2=30, y2=40)
        assert b.width == 20
        assert b.height == 30
        assert b.area == 600

    def test_clamp(self):
        b = Box(cls="apple", x1=-5, y1=-5, x2=500, y2=500)
        c = b.clamp(100, 100)
        assert (c.x1, c.y1, c.x2, c.y2) == (0, 0, 100, 100)

    def test_to_yolo_rejects_bad_dims(self):
        with pytest.raises(ValueError):
            Box(cls="apple", x1=0, y1=0, x2=1, y2=1).to_yolo(0, 100)


class TestManifest:
    def test_add_and_persist(self, tmp_path):
        m = Manifest(tmp_path / "manifest.jsonl")
        m.add(Record(id="000001", image="images/1.jpg", width=100, height=100))
        m.save()

        loaded = Manifest(tmp_path / "manifest.jsonl").load()
        assert len(loaded) == 1
        assert loaded.get("000001").width == 100

    def test_next_id_increments(self, tmp_path):
        m = Manifest(tmp_path / "m.jsonl")
        assert m.next_id() == "000001"
        m.add(Record(id="000001", image="a", width=1, height=1))
        assert m.next_id() == "000002"

    def test_next_id_fills_gap(self, tmp_path):
        m = Manifest(tmp_path / "m.jsonl")
        m.add(Record(id="000001", image="a", width=1, height=1))
        m.add(Record(id="000003", image="b", width=1, height=1))
        assert m.next_id() == "000002"

    def test_duplicate_id_rejected(self, tmp_path):
        m = Manifest(tmp_path / "m.jsonl")
        m.add(Record(id="000001", image="a", width=1, height=1))
        with pytest.raises(ValueError):
            m.add(Record(id="000001", image="b", width=1, height=1))

    def test_boxes_survive_roundtrip(self, tmp_path):
        m = Manifest(tmp_path / "m.jsonl")
        rec = Record(id="000001", image="images/1.jpg", width=200, height=200)
        rec.boxes = [Box(cls="apple", x1=1, y1=2, x2=30, y2=40, conf=0.9)]
        m.add(rec)
        m.save()

        loaded = Manifest(tmp_path / "m.jsonl").load()
        b = loaded.get("000001").boxes[0]
        assert b.cls == "apple"
        assert b.conf == pytest.approx(0.9)

    def test_summary(self, tmp_path):
        m = Manifest(tmp_path / "m.jsonl")
        r1 = Record(id="000001", image="a", width=10, height=10, status="labeled")
        r1.boxes = [Box(cls="apple", x1=0, y1=0, x2=5, y2=5)]
        r2 = Record(id="000002", image="b", width=10, height=10, status="labeled")
        r3 = Record(id="000003", image="c", width=10, height=10,
                    reject_reason="blurry_10.0")
        for r in (r1, r2, r3):
            m.add(r)

        s = m.summary()
        assert s["total"] == 3
        assert s["usable"] == 2
        assert s["rejected"] == 1
        assert s["positives"] == 1
        assert s["negatives"] == 1
        assert s["boxes_per_class"] == {"apple": 1}
        assert "blurry_10.0" in s["reject_reasons"]


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"objects": []}') == {"objects": []}

    def test_fenced(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
        assert extract_json(text) == {"a": 1}

    def test_with_prose(self):
        text = 'I found two apples. {"objects": [{"label": "apple"}]} Hope this helps!'
        out = extract_json(text)
        assert out["objects"][0]["label"] == "apple"

    def test_trailing_comma(self):
        assert extract_json('{"a": 1,}') == {"a": 1}

    def test_bare_list(self):
        assert extract_json('[{"label": "apple"}]') == [{"label": "apple"}]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            extract_json("   ")

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            extract_json("I cannot help with that.")


class TestNormalizeObjects:
    def test_objects_key(self):
        payload = {
            "objects": [
                {"label": "apple", "confidence": 0.9, "bbox": [1, 2, 30, 40]},
            ]
        }
        objs = normalize_objects(payload)
        assert len(objs) == 1
        assert objs[0].label == "apple"
        assert objs[0].bbox == [1.0, 2.0, 30.0, 40.0]

    def test_alternate_keys(self):
        payload = {"detections": [{"class": "lemon", "score": 0.8, "box": [1, 1, 5, 5]}]}
        objs = normalize_objects(payload)
        assert objs[0].label == "lemon"
        assert objs[0].confidence == pytest.approx(0.8)

    def test_percent_confidence(self):
        objs = normalize_objects([{"label": "apple", "confidence": 92, "bbox": [1, 1, 5, 5]}])
        assert objs[0].confidence == pytest.approx(0.92)

    def test_single_object_dict(self):
        objs = normalize_objects({"label": "apple", "bbox": [1, 1, 5, 5]})
        assert len(objs) == 1

    def test_bbox_as_dict(self):
        payload = [{"label": "apple", "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}]
        objs = normalize_objects(payload)
        assert objs[0].bbox == [1.0, 2.0, 3.0, 4.0]

    def test_bbox_as_xywh_dict(self):
        payload = [{"label": "apple", "bbox": {"x": 10, "y": 20, "w": 5, "h": 6}}]
        objs = normalize_objects(payload)
        assert objs[0].bbox == [10.0, 20.0, 15.0, 26.0]

    def test_bbox_as_string(self):
        payload = [{"label": "apple", "bbox": "1,2,30,40"}]
        objs = normalize_objects(payload)
        assert objs[0].bbox == [1.0, 2.0, 30.0, 40.0]

    def test_drops_items_without_label(self):
        payload = [{"confidence": 0.9, "bbox": [1, 1, 5, 5]}]
        assert normalize_objects(payload) == []

    def test_drops_items_without_bbox(self):
        assert normalize_objects([{"label": "apple"}]) == []

    def test_empty_objects(self):
        assert normalize_objects({"objects": []}) == []

    def test_json_string_roundtrip(self):
        text = '{"objects": [{"label": "lemon", "confidence": 0.7, "bbox": [0, 0, 10, 10]}]}'
        objs = normalize_objects(json.loads(text))
        assert objs[0].label == "lemon"
