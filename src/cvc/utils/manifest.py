"""Manifest：整条流水线唯一的中间数据契约。

每个阶段只做两件事：读 manifest → 改自己负责的字段 → 写回 manifest。
因此任何一步都可以中断、重跑、替换实现，而不会破坏其他阶段。

存储格式：JSONL（一行一条记录），便于 grep、便于流式处理、git diff 友好。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# 记录状态机：ingested → cleaned → labeled → exported
STATUS_ORDER = ("ingested", "cleaned", "labeled", "exported")


@dataclass
class Box:
    """一个标注框，坐标统一为**像素**（绝对坐标系，左上原点）。"""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    source: str = "auto"  # auto | human | imported

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, img_w: int, img_h: int) -> Box:
        """把框裁到图像范围内，并修正 x1<=x2 / y1<=y2。"""
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return Box(
            cls=self.cls,
            x1=max(0.0, min(x1, img_w)),
            y1=max(0.0, min(y1, img_h)),
            x2=max(0.0, min(x2, img_w)),
            y2=max(0.0, min(y2, img_h)),
            conf=self.conf,
            source=self.source,
        )

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """转成 YOLO 的 (cx, cy, w, h)，全部归一化到 0~1。"""
        if img_w <= 0 or img_h <= 0:
            raise ValueError("图像宽高必须为正")
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w = self.width / img_w
        h = self.height / img_h
        return cx, cy, w, h

    @classmethod
    def from_yolo(
        cls, line: str, img_w: int, img_h: int, names: list[str]
    ) -> Box:
        cid, cx, cy, w, h = (float(v) for v in line.split()[:5])
        name = names[int(cid)] if int(cid) < len(names) else str(int(cid))
        bw, bh = w * img_w, h * img_h
        return cls(
            cls=name,
            x1=cx * img_w - bw / 2,
            y1=cy * img_h - bh / 2,
            x2=cx * img_w + bw / 2,
            y2=cy * img_h + bh / 2,
            conf=1.0,
            source="imported",
        )


@dataclass
class Record:
    """一条样本记录。"""

    id: str
    image: str  # 相对 workspace 的路径，例如 images/000001.jpg
    width: int
    height: int

    source: str = ""  # 原始文件绝对路径
    sha256: str = ""
    phash: str = ""
    blur: float | None = None

    status: str = "ingested"
    reject_reason: str | None = None
    boxes: list[Box] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- 便捷属性 ----

    @property
    def is_usable(self) -> bool:
        return self.reject_reason is None

    @property
    def has_target(self) -> bool:
        return len(self.boxes) > 0

    @property
    def needs_review(self) -> bool:
        return "needs_review" in self.flags

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    def unflag(self, name: str) -> None:
        if name in self.flags:
            self.flags.remove(name)

    # ---- 序列化 ----

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["boxes"] = [asdict(b) for b in self.boxes]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        boxes = [Box(**b) for b in d.pop("boxes", [])]
        return cls(boxes=boxes, **d)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    """JSONL manifest 的读写封装。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[Record] = []
        self._index: dict[str, Record] = {}

    # ---- IO ----

    def load(self) -> Manifest:
        self.records.clear()
        self._index.clear()
        if not self.path.exists():
            return self
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = Record.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"{self.path}:{lineno} 解析失败：{exc}") from exc
                if rec.id in self._index:
                    # manifest 是给人看的 JSONL，可能被手工编辑过。
                    # 重复 id 会让 _index 静默指向最后一条，前面那条永远改不到。
                    raise ValueError(
                        f"{self.path}:{lineno} 记录 id 重复：{rec.id}。"
                        "请手工修正后重试。"
                    )
                self.records.append(rec)
                self._index[rec.id] = rec
        return self

    def save(self) -> None:
        """原子写：先写临时文件再替换，避免中途崩溃损坏 manifest。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # ---- 访问 ----

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def get(self, rid: str) -> Record | None:
        return self._index.get(rid)

    def add(self, rec: Record) -> None:
        if rec.id in self._index:
            raise ValueError(f"记录 id 重复：{rec.id}")
        self.records.append(rec)
        self._index[rec.id] = rec

    def next_id(self) -> str:
        used = {int(r.id) for r in self.records if r.id.isdigit()}
        nxt = 1
        while nxt in used:
            nxt += 1
        return f"{nxt:06d}"

    def usable(self) -> list[Record]:
        return [r for r in self.records if r.is_usable]

    def by_status(self, status: str) -> list[Record]:
        return [r for r in self.records if r.status == status]

    # ---- 统计 ----

    def summary(self) -> dict[str, Any]:
        total = len(self.records)
        usable = sum(1 for r in self.records if r.is_usable)
        rejected = total - usable
        positives = sum(1 for r in self.records if r.is_usable and r.has_target)
        negatives = sum(1 for r in self.records if r.is_usable and not r.has_target)
        review = sum(1 for r in self.records if r.needs_review)

        per_class: dict[str, int] = {}
        for r in self.records:
            if not r.is_usable:
                continue
            for b in r.boxes:
                per_class[b.cls] = per_class.get(b.cls, 0) + 1

        reasons: dict[str, int] = {}
        for r in self.records:
            if r.reject_reason:
                reasons[r.reject_reason] = reasons.get(r.reject_reason, 0) + 1

        return {
            "total": total,
            "usable": usable,
            "rejected": rejected,
            "positives": positives,
            "negatives": negatives,
            "needs_review": review,
            "boxes_per_class": per_class,
            "reject_reasons": reasons,
        }

    def write_config_snapshot(self, yaml_text: str) -> None:
        """把本次运行的 recipe 存到 workspace，保证可复现。"""
        target = self.path.parent / "recipe.snapshot.yaml"
        target.write_text(yaml_text, encoding="utf-8")
