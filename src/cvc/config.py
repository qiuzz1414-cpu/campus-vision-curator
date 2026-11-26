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
        # ---- 类别 ----
        cats = self.label.categories
        if not cats:
            raise ConfigError("label.categories 不能为空，至少指定一个目标类别")
        if isinstance(cats, str):
            # YAML 里漏写 `-` 会得到字符串，例如 categories: "abc"
            # 若放过去，len(set("abc")) == len("abc") 会让重复检查失效，
            # 最终 class_names 变成 ['a', 'b', 'c']，训练标签整体错位。
            raise ConfigError(
                f"label.categories 必须是列表，当前是字符串 {cats!r}；"
                '正确写法：categories: ["apple", "lemon"]（每项前加 - 或写成行内列表）'
            )
        for c in cats:
            if not isinstance(c, str) or not c.strip():
                raise ConfigError(f"label.categories 中存在非法项：{c!r}，必须是非空字符串")
        if len(set(cats)) != len(cats):
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

        # ---- 调度参数下界：0 会造成挂死或无重试，且都不会报错 ----
        if not isinstance(self.label.concurrency, int) or self.label.concurrency < 1:
            raise ConfigError(
                f"label.concurrency 必须 >= 1，当前 {self.label.concurrency!r}；"
                "0 会让 Semaphore(0) 永远拿不到信号量，任务直接挂死"
            )
        if not isinstance(self.label.retries, int) or self.label.retries < 1:
            raise ConfigError(
                f"label.retries 必须 >= 1，当前 {self.label.retries!r}；"
                "0 表示一次都不尝试，只会留下一条无信息的失败记录"
            )
        if not 0.0 <= self.label.min_conf <= 1.0:
            raise ConfigError(f"label.min_conf 必须在 0~1 之间，当前 {self.label.min_conf}")
        if not 0.0 <= self.label.review_conf <= 1.0:
            raise ConfigError(
                f"label.review_conf 必须在 0~1 之间，当前 {self.label.review_conf}"
            )

        # ---- 划分 ----
        split = self.export.split
        if not isinstance(split, dict) or not split:
            raise ConfigError("export.split 必须是形如 {train: 0.8, val: 0.2} 的映射")
        unknown_keys = set(split) - {"train", "val", "test"}
        if unknown_keys:
            raise ConfigError(
                f"export.split 中存在未知划分：{sorted(unknown_keys)}，只支持 train / val / test"
            )
        # train 与 val 是下游硬依赖（stratified_split 与 data.yaml 都会直接用），
        # 缺了不是"少一个划分"而是 KeyError，所以必须前置校验。
        for required in ("train", "val"):
            if required not in split:
                raise ConfigError(
                    f"export.split 缺少 `{required}`；至少需要 train 和 val"
                    "（test 可选）。YOLO 训练必须有验证集才能算 mAP。"
                )
        for name, ratio in split.items():
            if not isinstance(ratio, (int, float)):
                raise ConfigError(f"export.split.{name} 必须是数字，当前 {ratio!r}")
            if ratio < 0:
                raise ConfigError(f"export.split.{name} 不能为负")
        total = sum(split.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"export.split 比例之和应为 1.0，当前为 {total}")

        if not 0.0 <= self.export.negative_ratio <= 1.0:
            raise ConfigError(
                f"export.negative_ratio 必须在 0~1 之间，当前 {self.export.negative_ratio}"
            )

        # ---- 训练 ----
        for field_name in ("epochs", "imgsz", "batch"):
            value = getattr(self.train, field_name)
            if not isinstance(value, int) or value < 1:
                raise ConfigError(f"train.{field_name} 必须是 >= 1 的整数，当前 {value!r}")

        # ---- 输入 ----
        if not isinstance(self.input.paths, list):
            raise ConfigError("input.paths 必须是列表，例如 paths: [raw]")
        if self.input.max_side and (
            not isinstance(self.input.max_side, int) or self.input.max_side < 64
        ):
            raise ConfigError(
                f"input.max_side 必须是 >= 64 的整数或 0（不缩放），当前 {self.input.max_side!r}"
            )



def _coerce_scalar(value: Any, tp: Any, path: str) -> Any:
    """对叶子值做类型检查。

    YAML 对类型很宽容：`max_side: "1600"` 会被读成字符串，然后在下游某个
    `max(w, h) > max_side` 处抛 `TypeError: '>' not supported`——错误信息离
    出错的地方有十万八千里，学生根本定位不了。这里把类型问题挡在配置层。
    """
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} 必须是 true/false，当前 {value!r}")
        return value
    if tp in (int, float):
        # YAML 里 1 和 1.0 可能都出现，整数位允许浮点但必须无损
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} 必须是数字，当前 {value!r}（类型 {type(value).__name__}）")
        if tp is int and float(value) != int(value):
            raise ConfigError(f"{path} 必须是整数，当前 {value!r}")
        return int(value) if tp is int else float(value)
    if tp is str:
        if value is None:
            raise ConfigError(
                f"{path} 不能为空（YAML 里单独的 `~` 或留空会被解析成 null）"
            )
        if not isinstance(value, str):
            raise ConfigError(f"{path} 必须是字符串，当前 {value!r}")
        return value
    return value


def _build(cls: type, data: Any, path: str = "") -> Any:
    """按 dataclass 定义递归构建嵌套配置，容忍缺字段（用默认值）。"""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__} 期望字典，收到 {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__} 中存在未知字段：{sorted(unknown)}；"
            f"可用字段：{sorted(known)}"
        )

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        tp = hints.get(f.name)
        value = data[f.name]
        child_path = f"{path}.{f.name}" if path else f.name
        if dataclasses.is_dataclass(tp):
            kwargs[f.name] = _build(tp, value, child_path)
        else:
            kwargs[f.name] = _coerce_scalar(value, tp, child_path)
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


def dump_config(cfg: Config, mask_secrets: bool = True) -> str:
    """把配置回写为 YAML 文本（用于日志留痕 / 复现实验）。

    默认脱敏 api_key：这份快照会落在 `workspace/recipe.snapshot.yaml`，
    而 workspace 经常被打包上交或同步到网盘。真实密钥不该跟着数据走。
    """

    def _plain(obj: Any, key: str = "") -> Any:
        if mask_secrets and key == "api_key" and isinstance(obj, str) and obj not in ("", "EMPTY"):
            return f"***{obj[-4:]}" if len(obj) > 4 else "***"
        if dataclasses.is_dataclass(obj):
            return {f.name: _plain(getattr(obj, f.name), f.name) for f in dataclasses.fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _plain(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_plain(v) for v in obj]
        return obj

    data = _plain(cfg)
    data.pop("recipe_path", None)
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
