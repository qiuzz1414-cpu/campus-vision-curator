"""命令行入口。

设计原则：每个阶段一条命令，可以单独跑、可以反复跑（幂等），
也可以用 `cvc run` 一口气跑完整条流水线。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml

from . import __version__
from .config import Config, ConfigError, dump_config, load_config
from .utils.logging import banner, get_logger
from .utils.manifest import Manifest

app = typer.Typer(
    name="cvc",
    help="campus-vision-curator — 从原始照片到 YOLO 可训练数据集的策展流水线",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)

log = get_logger("cli")

RECIPE_OPT = typer.Option(
    "recipes/fruit_picking.yaml", "--recipe", "-r", help="recipe YAML 路径"
)
FORCE_OPT = typer.Option(False, "--force", "-f", help="忽略已完成状态，强制重跑")


# ---------------------------------------------------------------- 公共


def _load(recipe: str, ensure: bool = True) -> tuple[Config, Manifest]:
    """加载配置与 manifest。

    ensure=False 用于 `clean` 这类删除类命令：**绝不能**在守卫之前创建目录，
    否则 ensure_dirs() 会亲手造出"这看起来像个 workspace"的证据，
    让 "workspace: ." 这种配置骗过保护逻辑。
    """
    try:
        cfg = load_config(recipe)
    except ConfigError as exc:
        typer.secho(f"配置错误：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except yaml.YAMLError as exc:
        # YAML 语法错误很常见（缩进、制表符、未加引号的冒号），
        # 不打成友好提示的话用户只会看到一坨 ParserError 堆栈
        typer.secho(
            f"recipe 语法错误：{recipe}\n{exc}\n"
            "提示：YAML 只允许空格缩进，不能用 Tab；值里含冒号时要加引号。",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    if ensure:
        try:
            cfg.ensure_dirs()
        except OSError as exc:
            typer.secho(
                f"无法创建 workspace 目录：{cfg.workspace_path}\n  {exc}\n"
                "检查：路径是否存在、是否有写权限、磁盘是否已满。",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from exc

    try:
        manifest = Manifest(cfg.manifest_path).load()
    except ValueError as exc:
        typer.secho(
            f"manifest 解析失败：{exc}\n"
            "提示：manifest.jsonl 一行一条 JSON。手工编辑后请确保每行都合法。",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    # 首次运行时留一份配置快照，保证结果可复现（api_key 会被脱敏）
    if ensure:
        snapshot = cfg.workspace_path / "recipe.snapshot.yaml"
        if not snapshot.exists():
            try:
                snapshot.write_text(dump_config(cfg), encoding="utf-8")
            except OSError as exc:
                # 快照只是便利功能，写不进去不该让整个命令挂掉
                log.warning(f"无法写入配置快照（不影响本次运行）：{exc}")

    return cfg, manifest


def _print_summary(cfg: Config, manifest: Manifest) -> None:
    s = manifest.summary()
    rows = [
        ("样本总数", s["total"]),
        ("可用", s["usable"]),
        ("已剔除", s["rejected"]),
        ("正样本", s["positives"]),
        ("负样本", s["negatives"]),
        ("待复核", s["needs_review"]),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")
    if s["boxes_per_class"]:
        print("  各类别框数：")
        for cls, n in sorted(s["boxes_per_class"].items()):
            print(f"    {cls.ljust(width)}  {n}")
    if s["reject_reasons"]:
        print("  剔除原因：")
        for reason, n in sorted(s["reject_reasons"].items()):
            print(f"    {reason.ljust(width)}  {n}")


# ---------------------------------------------------------------- 命令


@app.command()
def version() -> None:
    """显示版本号。"""
    print(f"campus-vision-curator {__version__}")


def _find_asset(kind: str, name: str) -> Path | None:
    """定位随包分发的资源文件（recipe 模板、辅助脚本）。

    两种安装形态要找的位置不同：
    - wheel 安装：随包打入 `cvc/_<kind>/`
    - 源码 / editable 安装：仓库根的 `<kind>/`
    """
    here = Path(__file__).resolve().parent
    for candidate in (
        here / f"_{kind}" / name,
        here.parent.parent / kind / name,
    ):
        if candidate.exists():
            return candidate
    return None


def _find_template(name: str = "fruit_picking.yaml") -> Path | None:
    return _find_asset("recipes", name)


@app.command()
def init(
    recipe: str = RECIPE_OPT,
    force: bool = typer.Option(False, "--force", help="覆盖已存在的 recipe"),
) -> None:
    """初始化工作区与 recipe 模板。"""
    target = Path(recipe)
    if target.exists() and not force:
        typer.secho(f"recipe 已存在：{target}（加 --force 覆盖）", fg=typer.colors.YELLOW)
    else:
        # 注意这里必须是 else 而不是 elif not target.exists()：
        # 否则 `--force` 在文件已存在时两个分支都不进，命令静默变成空操作。
        target.parent.mkdir(parents=True, exist_ok=True)
        template = _find_template()
        if template is not None:
            shutil.copy(template, target)
            typer.secho(f"已从模板创建 recipe：{target}", fg=typer.colors.GREEN)
        else:
            target.write_text(
                "project: my-dataset\n"
                "workspace: workspace\n"
                "input:\n  paths: [raw]\n"
                "label:\n"
                "  categories: [cat, dog]   # 改成你自己的类别\n"
                "  distractors: []\n",
                encoding="utf-8",
            )
            typer.secho(
                f"未找到内置模板，已生成最小 recipe：{target}", fg=typer.colors.YELLOW
            )

    cfg, _ = _load(str(target))
    cfg.ensure_dirs()
    typer.secho(f"工作区：{cfg.workspace_path}", fg=typer.colors.GREEN)
    typer.echo("\n下一步：把照片放进 recipe 的 input.paths 目录，然后执行 cvc ingest")


def _doctor_vlm(cfg: Config) -> None:
    """VLM 连通性检查（mock 后端不会走到这里）。"""
    from .utils.vlm import list_models

    try:
        models = list_models(cfg.label.base_url, cfg.label.api_key)
    except Exception as exc:
        print(f"  ❌ 无法连接 {cfg.label.base_url}：{exc}")
        script = _find_asset("scripts", "serve_vllm.sh")
        if script is not None:
            print(f"     本地：bash {script}")
        else:
            print("     本地：bash scripts/serve_vllm.sh（在仓库目录下）")
        print("     云端：把 recipe 的 base_url / api_key 换成服务商地址")
        return

    print(f"  ✅ {cfg.label.base_url}")
    print(f"     可用模型：{', '.join(models[:8]) or '(空)'}")
    if cfg.label.model not in models:
        print(f"  ⚠️  recipe 指定的 `{cfg.label.model}` 不在列表中，请核对模型名")

    if cfg.label.bbox_format == "auto":
        print("  ⚠️  bbox_format=auto：坐标格式靠启发式猜测，个别图可能猜错（表现为框整体偏移）")
        print("     建议按模型族写死：Qwen2.5-VL→absolute，Qwen3-VL→norm1000")


@app.command()
def doctor(recipe: str = RECIPE_OPT) -> None:
    """自检：依赖、图片目录、VLM 服务连通性。"""
    cfg, _ = _load(recipe)

    print("\n[依赖]")
    for mod, hint in (
        ("PIL", "pillow"),
        ("cv2", "opencv-python-headless"),
        ("numpy", "numpy"),
        ("imagehash", "imagehash"),
        ("yaml", "pyyaml"),
        ("openai", "openai"),
        ("typer", "typer"),
    ):
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except ImportError:
            print(f"  ❌ {mod}（pip install {hint}）")

    for mod, extra in (("ultralytics", "train"), ("torch", "clip")):
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except ImportError:
            print(f"  ⚪ {mod} 未安装（可选：pip install -e '.[{extra}]'）")

    print("\n[输入目录]")
    from .stages.s1_ingest import IMAGE_EXT, VIDEO_EXT

    supported = IMAGE_EXT | VIDEO_EXT
    for p in cfg.input.paths:
        path = Path(p).expanduser()
        if not path.exists():
            print(f"  ❌ 不存在：{path}")
            continue
        # 用 ingest 阶段同一份扩展名集合，避免 doctor 说"有 0 个文件"
        # 而 ingest 实际能找到素材
        n = sum(1 for f in path.rglob("*") if f.is_file() and f.suffix.lower() in supported)
        print(f"  ✅ {path}（{n} 个素材文件）")

    print("\n[标注后端]")
    if cfg.label.backend == "mock":
        # mock 后端完全离线，再去连 VLM 只会给出误导性的红叉
        print("  ⚪ backend=mock：使用离线假框，跳过 VLM 连通性检查")
        print("     仅用于验证流水线，训练前必须切回 openai_compat")
    else:
        print(f"  backend=openai_compat · 模型 {cfg.label.model}")
        _doctor_vlm(cfg)

    print("\n[类别]")
    print(f"  {', '.join(cfg.class_names) or '(未配置)'}")
    if cfg.label.distractors:
        print(f"  需忽略：{', '.join(cfg.label.distractors)}")


@app.command()
def ingest(recipe: str = RECIPE_OPT) -> None:
    """阶段一：接入原始照片/视频，统一尺寸与方向。"""
    from .stages import s1_ingest

    cfg, manifest = _load(recipe)
    s1_ingest.run(cfg, manifest)


@app.command()
def dedup(recipe: str = RECIPE_OPT, force: bool = FORCE_OPT) -> None:
    """阶段二：清晰度过滤 + 感知哈希去重。"""
    from .stages import s2_dedup

    cfg, manifest = _load(recipe)
    s2_dedup.run(cfg, manifest, force=force)


@app.command()
def label(
    recipe: str = RECIPE_OPT,
    force: bool = FORCE_OPT,
    limit: int = typer.Option(0, "--limit", help="只标前 N 张（用于先试跑几张）"),
) -> None:
    """阶段三：调用 VLM 自动标注。"""
    from .stages import s3_autolabel

    cfg, manifest = _load(recipe)
    s3_autolabel.run(cfg, manifest, force=force, limit=limit or None)


@app.command()
def visualize(recipe: str = RECIPE_OPT) -> None:
    """阶段四 a：画框 + 生成 HTML 质检报告。"""
    from .stages import s4_visualize

    cfg, manifest = _load(recipe)
    result = s4_visualize.run(cfg, manifest)
    if result.get("report"):
        typer.secho(f"\n用浏览器打开：{result['report']}", fg=typer.colors.GREEN)


@app.command("review-export")
def review_export(
    recipe: str = RECIPE_OPT,
    all_samples: bool = typer.Option(
        False, "--all", help="导出全部样本，而不只是被标记的"
    ),
) -> None:
    """阶段四 b：导出复核包（YOLO txt），交给标注工具人工修正。"""
    from .stages import s4_visualize

    cfg, manifest = _load(recipe)
    out = s4_visualize.export_review(cfg, manifest, only_flagged=not all_samples)
    typer.secho(f"\n复核包：{out}", fg=typer.colors.GREEN)
    typer.echo("用标注工具改完 review/labels/*.txt 后执行：cvc review-import")


@app.command("review-import")
def review_import(recipe: str = RECIPE_OPT) -> None:
    """阶段四 c：把人工修正过的标签回写 manifest。"""
    from .stages import s4_visualize

    cfg, manifest = _load(recipe)
    s4_visualize.import_review(cfg, manifest)


@app.command("export")
def export_cmd(
    recipe: str = RECIPE_OPT,
    keep: bool = typer.Option(False, "--keep", help="保留旧 dataset 目录（默认清空重建）"),
) -> None:
    """阶段五：导出 YOLO 数据集 + 统计表。"""
    from .stages import s5_export

    cfg, manifest = _load(recipe)
    stats = s5_export.run(cfg, manifest, clean=not keep)
    if stats.get("total_images"):
        print()
        _print_summary(cfg, manifest)


@app.command()
def train(
    recipe: str = RECIPE_OPT,
    epochs: int = typer.Option(0, "--epochs", help="覆盖 recipe 里的 epochs"),
    model: str = typer.Option("", "--model", help="覆盖基础模型，如 yolov8s.pt"),
    batch: int = typer.Option(0, "--batch", help="覆盖 batch size"),
) -> None:
    """阶段六：训练 YOLO 并汇总指标。"""
    from .stages import s6_train

    cfg, _ = _load(recipe)
    overrides: dict = {}
    # 覆盖值必须同时回写进 cfg：summary.md 是从 cfg.train.* 渲染的，
    # 只放进 overrides 会让报告写出 recipe 默认值，与实际训练参数不符。
    if epochs:
        overrides["epochs"] = epochs
        cfg.train.epochs = epochs
    if model:
        overrides["model"] = model
        cfg.train.model = model
    if batch:
        overrides["batch"] = batch
        cfg.train.batch = batch

    s6_train.run(cfg, overrides)


@app.command()
def run(
    recipe: str = RECIPE_OPT,
    skip_train: bool = typer.Option(False, "--skip-train", help="跑到导出就停，不训练"),
    skip_label: bool = typer.Option(False, "--skip-label", help="跳过 VLM 标注（已有标注时用）"),
) -> None:
    """一键跑完整条流水线。"""
    from .stages import s1_ingest, s2_dedup, s3_autolabel, s4_visualize, s5_export, s6_train

    cfg, manifest = _load(recipe)

    s1_ingest.run(cfg, manifest)
    s2_dedup.run(cfg, manifest)

    if skip_label:
        log.info("按 --skip-label 跳过阶段三")
    else:
        s3_autolabel.run(cfg, manifest)

    s4_visualize.run(cfg, manifest)
    s5_export.run(cfg, manifest)

    if skip_train:
        log.info("按 --skip-train 跳过阶段六")
    else:
        s6_train.run(cfg)

    banner("流水线结束")
    _print_summary(cfg, manifest)
    typer.secho(
        f"\n质检报告：{cfg.workspace_path / 'report.html'}\n"
        f"数据集：  {cfg.dataset_dir}\n"
        f"统计表：  {cfg.dataset_dir / 'stats.md'}",
        fg=typer.colors.GREEN,
    )


@app.command()
def status(recipe: str = RECIPE_OPT) -> None:
    """查看当前 manifest 状态。"""
    cfg, manifest = _load(recipe)
    banner(f"状态 · {cfg.project} · {cfg.workspace_path}")

    from collections import Counter

    by_status = Counter(r.status for r in manifest)
    print("  阶段分布：")
    for st in ("ingested", "cleaned", "labeled", "exported"):
        print(f"    {st.ljust(10)} {by_status.get(st, 0)}")
    print()
    _print_summary(cfg, manifest)


@app.command("clean")
def clean_cmd(
    recipe: str = RECIPE_OPT,
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
) -> None:
    """清空工作区（保留 recipe）。危险操作。"""
    # ensure=False：守卫必须在任何创建目录的动作之前执行
    cfg, _ = _load(recipe, ensure=False)
    target = cfg.workspace_path

    def _refuse(reason: str) -> None:
        typer.secho(f"拒绝执行：{reason}\n未删除任何文件。", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # ---- 守卫 1：危险锚点 ----
    if target == Path(target.anchor):
        _refuse(f"workspace 指向文件系统根目录 {target}")
    if target == Path.home().resolve():
        _refuse(f"workspace 指向家目录 {target}")

    # ---- 守卫 2：不能是当前目录或其祖先 ----
    # `workspace: .` / `workspace: ..` 会把项目本身删掉。
    # 注意：这里不能靠"目录里有 images/ 说明是 workspace"来判断——
    # 之前的实现正是这么做的，结果 ensure_dirs() 自己创建了 images/，
    # 让 "workspace: ." 顺利通过了检查。
    cwd = Path.cwd().resolve()
    if cwd == target or cwd.is_relative_to(target):
        _refuse(
            f"workspace ({target}) 是当前工作目录或其上级目录。\n"
            "请先 cd 到 workspace 之外的目录，再执行 cvc clean。"
        )

    # ---- 守卫 3：不能是 recipe 文件所在目录或它的祖先 ----
    if cfg.recipe_path is not None:
        recipe_dir = cfg.recipe_path.parent.resolve()
        if recipe_dir == target or recipe_dir.is_relative_to(target):
            _refuse(f"workspace ({target}) 包含 recipe 文件本身（{cfg.recipe_path}）")

    # ---- 守卫 4：真的是本项目的工作区 ----
    if not (target / "manifest.jsonl").exists():
        _refuse(
            f"{target} 下没有 manifest.jsonl，看起来不是本项目的 workspace。\n"
            "如果你确实要删，请手工确认路径后删除。"
        )

    if not yes:
        typer.confirm(f"将删除 {target} 内所有产物，确认？", abort=True)
    shutil.rmtree(target)
    typer.secho(f"已删除 {target}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
