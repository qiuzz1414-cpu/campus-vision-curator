"""Recipe（YAML 配置）加载与校验。

设计原则：用户只改 YAML，不改代码。
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Recipe 配置错误。"""


@dataclass
class InputCfg:
    """阶段一：原始素材接入。"""

    paths: list[str] = field(default_factory=lambda: ["raw"])
    video_fps: float = 1.0
    max_side: int = 1600
    jpeg_quality: int = 95
    recursive: bool = True


@dataclass
class ClipCfg:
    """CLIP 语义去重（可选，需要额外安装）。"""

    enabled: bool = False
    threshold: float = 0.95
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"


@dataclass
class FilterCfg:
    """阶段二：质量过滤与去重。"""

    blur_threshold: float = 60.0
    phash_distance: int = 6
    min_side: int = 200
    clip: ClipCfg = field(default_factory=ClipCfg)


@dataclass
class LabelCfg:
    """阶段三：VLM 自动标注。"""

    backend: str = "openai_compat"  # openai_compat | mock
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen2.5-VL-7B-Instruct"
    bbox_format: str = "auto"  # auto | absolute | norm1000 | norm1
    json_mode: bool = False
    concurrency: int = 4
    timeout: float = 120.0
    retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 1024
    image_max_side: int = 1024
    categories: list[str] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
    min_conf: float = 0.25
    review_conf: float = 0.6
    prompt_extra: str = ""


@dataclass
class ExportCfg:
    """阶段五：导出 YOLO 数据集。"""

    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.15, "test": 0.05}
    )
    seed: int = 42
    negative_ratio: float = 0.1
    copy_images: bool = True


@dataclass
class TrainCfg:
    """阶段六：YOLO 训练。"""

    enabled: bool = True
    model: str = "yolov8n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    patience: int = 30
    workers: int = 4


@dataclass
class Config:
    project: str = "campus-vision"
    workspace: str = "workspace"
    input: InputCfg = field(default_factory=InputCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    label: LabelCfg = field(default_factory=LabelCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # 运行时填充，不来自 YAML
    recipe_path: Path | None = None

    # ---- 派生路径 ----

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    @property
    def manifest_path(self) -> Path:
        return self.workspace_path / "manifest.jsonl"

    @property
    def images_dir(self) -> Path:
        return self.workspace_path / "images"

    @property
    def rejects_dir(self) -> Path:
        return self.workspace_path / "rejects"

    @property
    def overlays_dir(self) -> Path:
        return self.workspace_path / "overlays"

    @property
    def dataset_dir(self) -> Path:
        return self.workspace_path / "dataset"

    @property
    def review_dir(self) -> Path:
        return self.workspace_path / "review"

    @property
    def class_names(self) -> list[str]:
        return list(self.label.categories)

    def ensure_dirs(self) -> None:
        for d in (
            self.workspace_path,
            self.images_dir,
            self.rejects_dir,
            self.overlays_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.label.categories:
            raise ConfigError("label.categories 不能为空，至少指定一个目标类别")
        if len(set(self.label.categories)) != len(self.label.categories):
            raise ConfigError("label.categories 存在重复类别名")
        if self.label.backend not in {"openai_compat", "mock"}:
            raise ConfigError(
                f"label.backend 非法：{self.label.backend}，可选 openai_compat / mock"
            )
        if self.label.bbox_format not in {"auto", "absolute", "norm1000", "norm1"}:
            raise ConfigError(
                f"label.bbox_format 非法：{self.label.bbox_format}，"
                "可选 auto / absolute / norm1000 / norm1"
            )
        total = sum(self.export.split.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"export.split 比例之和应为 1.0，当前为 {total}")
        for name, ratio in self.export.split.items():
            if ratio < 0:
                raise ConfigError(f"export.split.{name} 不能为负")


def _build(cls: type, data: Any) -> Any:
    """按 dataclass 定义递归构建嵌套配置，容忍缺字段（用默认值）。"""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} 期望字典，收到 {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{cls.__name__} 中存在未知字段：{sorted(unknown)}；"
            f"可用字段：{sorted(known)}"
        )

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        tp = hints.get(f.name)
        value = data[f.name]
        if dataclasses.is_dataclass(tp):
            kwargs[f.name] = _build(tp, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """从 YAML 读取 recipe。相对路径的 workspace / input.paths 以当前工作目录为基准。"""
    recipe_path = Path(path).expanduser()
    if not recipe_path.exists():
        raise ConfigError(f"找不到 recipe 文件：{recipe_path}")

    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("recipe 顶层必须是映射（key: value）")

    cfg = _build(Config, raw)
    cfg.recipe_path = recipe_path.resolve()
    cfg.validate()
    return cfg


def dump_config(cfg: Config) -> str:
    """把配置回写为 YAML 文本（用于日志留痕 / 复现实验）。"""

    def _plain(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_plain(v) for v in obj]
        return obj

    data = _plain(cfg)
    data.pop("recipe_path", None)
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
