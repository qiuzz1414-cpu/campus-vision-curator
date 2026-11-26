"""针对 OCR 审查发现的缺陷写的回归测试。

每一条都应该在修复前的代码上失败。命名统一为 `test_<问题>`，
方便以后有人改回去时立刻发现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from cvc.config import ConfigError, dump_config, load_config
from cvc.stages.s1_ingest import _skip, collect_files
from cvc.stages.s4_visualize import import_review
from cvc.utils.bbox_convert import detect_format, to_pixels
from cvc.utils.manifest import Box, Manifest, Record

# ---------------------------------------------------------------- 配置层


class TestConfigValidation:
    """H: `categories` 写成字符串会被当成多类别；split 缺键会 KeyError。"""

    def _cfg(self, tmp_path: Path, body: str):
        p = tmp_path / "r.yaml"
        p.write_text(body, encoding="utf-8")
        return load_config(p)

    def test_string_categories_rejected(self, tmp_path):
        # 漏写 `-` 时 YAML 得到 "abc"，旧代码会通过校验并变成 ['a','b','c']
        with pytest.raises(ConfigError, match="必须是列表"):
            self._cfg(tmp_path, 'label:\n  categories: "apple"\n')

    def test_split_missing_train_rejected(self, tmp_path):
        # 旧代码只校验总和，{val: 0.5, test: 0.5} 能通过，然后在 s5 里 KeyError
        with pytest.raises(ConfigError, match="缺少 `train`"):
            self._cfg(
                tmp_path,
                "label:\n  categories: [a]\nexport:\n  split: {val: 0.5, test: 0.5}\n",
            )

    def test_split_missing_val_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="缺少 `val`"):
            self._cfg(
                tmp_path,
                "label:\n  categories: [a]\nexport:\n  split: {train: 1.0}\n",
            )

    def test_concurrency_zero_rejected(self, tmp_path):
        # Semaphore(0) 会静默挂死
        with pytest.raises(ConfigError, match="concurrency"):
            self._cfg(tmp_path, "label:\n  categories: [a]\n  concurrency: 0\n")

    def test_retries_zero_rejected(self, tmp_path):
        # 循环不执行 → 错误信息写成 "None"
        with pytest.raises(ConfigError, match="retries"):
            self._cfg(tmp_path, "label:\n  categories: [a]\n  retries: 0\n")

    def test_wrong_scalar_type_rejected(self, tmp_path):
        # "1600" 会在下游 max(w,h) > max_side 处抛 TypeError，错误点离得很远
        with pytest.raises(ConfigError, match="input.max_side"):
            self._cfg(tmp_path, 'label:\n  categories: [a]\ninput:\n  max_side: "1600"\n')

    def test_unknown_split_key_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="未知划分"):
            self._cfg(
                tmp_path,
                "label:\n  categories: [a]\nexport:\n  split: {train: 0.5, val: 0.3, dev: 0.2}\n",
            )


class TestSecretMasking:
    """security: api_key 曾被原样写进 workspace/recipe.snapshot.yaml。"""

    def test_api_key_masked_in_snapshot(self, tmp_path):
        p = tmp_path / "r.yaml"
        p.write_text(
            "label:\n  categories: [a]\n  api_key: sk-super-secret-key-1234\n",
            encoding="utf-8",
        )
        cfg = load_config(p)
        text = dump_config(cfg)

        assert "sk-super-secret-key-1234" not in text
        assert "***1234" in text

    def test_empty_key_not_mangled(self, tmp_path):
        p = tmp_path / "r.yaml"
        p.write_text("label:\n  categories: [a]\n  api_key: EMPTY\n", encoding="utf-8")
        assert "api_key: EMPTY" in dump_config(load_config(p))


# ---------------------------------------------------------------- 接入层


class TestSkipRule:
    """H: `_skip` 用绝对路径的每一段，祖先目录叫 images 就全跳过。"""

    def test_ancestor_named_images_does_not_skip(self, tmp_path):
        # 构造 /.../images/project/raw/a.jpg —— 祖先有 images，但文件不在 workspace 内
        raw = tmp_path / "images" / "project" / "raw"
        raw.mkdir(parents=True)
        f = raw / "a.jpg"
        f.write_bytes(b"x")

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        assert _skip(f, workspace) is False

    def test_hidden_file_skipped(self, tmp_path):
        f = tmp_path / ".DS_Store"
        f.write_bytes(b"x")
        assert _skip(f, tmp_path / "workspace") is True

    def test_workspace_images_skipped(self, tmp_path):
        workspace = tmp_path / "workspace"
        inner = workspace / "images" / "000001.jpg"
        inner.parent.mkdir(parents=True)
        inner.write_bytes(b"x")
        assert _skip(inner, workspace) is True

    def test_collect_finds_files_under_images_parent(self, tmp_path):
        # 端到端：旧实现这里会返回 0 个文件
        raw = tmp_path / "images" / "raw"
        raw.mkdir(parents=True)
        Image.new("RGB", (64, 64), (10, 20, 30)).save(raw / "a.jpg")
        Image.new("RGB", (64, 64), (40, 50, 60)).save(raw / "b.jpg")

        images, videos = collect_files([str(raw)], exclude_root=tmp_path / "workspace")
        assert len(images) == 2
        assert not videos


# ---------------------------------------------------------------- 划分层


class TestStratifiedCoverage:
    """[m37] 按主类别分桶无法保证每个类别都进 train。"""

    def _rec(self, rid: str, classes: list[str]) -> Record:
        r = Record(id=rid, image=f"images/{rid}.jpg", width=100, height=100, status="labeled")
        r.boxes = [Box(cls=c, x1=0, y1=0, x2=10, y2=10) for c in classes]
        return r

    def test_rare_class_forced_into_train(self):
        from cvc.stages.s5_export import stratified_split

        # 20 张苹果图，只有 1 张同时含柠檬 —— 柠檬从不成为主类别，
        # 因此旧实现（只按主类别分桶）完全依赖洗牌运气。
        # seed=15 是实测能在修复前代码上复现"train 里没有 lemon"的那一组。
        recs = [self._rec(f"{i:06d}", ["apple"]) for i in range(1, 21)]
        recs[0].boxes.append(Box(cls="lemon", x1=0, y1=0, x2=5, y2=5))

        out = stratified_split(recs, {"train": 0.8, "val": 0.15, "test": 0.05}, seed=15)

        train_classes = {b.cls for r in out["train"] for b in r.boxes}
        assert "lemon" in train_classes, "柠檬没有出现在 train 中，该类别将无法被识别"

    def test_train_only_split_does_not_reserve_val(self):
        from cvc.stages.s5_export import stratified_split

        recs = [self._rec(f"{i:06d}", ["apple"]) for i in range(1, 11)]
        out = stratified_split(recs, {"train": 1.0}, seed=1)
        assert len(out["train"]) == len(recs)


# ---------------------------------------------------------------- 复核闭环


class TestReviewLoop:
    """[m16] 只改类别被当 no-op；[m38] 空记录永远清不掉标志。"""

    def _setup(self, cfg, manifest):
        pass

    def test_class_only_change_is_applied(self, tmp_path, monkeypatch):
        from cvc.config import load_config

        recipe = tmp_path / "r.yaml"
        recipe.write_text(
            f"workspace: {tmp_path / 'ws'}\nlabel:\n  categories: [apple, lemon]\n",
            encoding="utf-8",
        )
        cfg = load_config(recipe)
        cfg.ensure_dirs()

        # 一张图，VLM 把它标成 apple（错误），复核员改成 lemon，框位置不动
        img = cfg.images_dir / "000001.jpg"
        Image.new("RGB", (100, 100), (1, 2, 3)).save(img)

        rec = Record(id="000001", image="images/000001.jpg", width=100, height=100, status="labeled")
        rec.boxes = [Box(cls="apple", x1=10, y1=10, x2=50, y2=50)]
        rec.flag("needs_review")

        manifest = Manifest(cfg.manifest_path)
        manifest.add(rec)
        manifest.save()

        # 构造复核包：同一坐标，但类别改成 lemon（class_id=1）
        lbl_dir = cfg.review_dir / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (lbl_dir / "000001.txt").write_text("1 0.300000 0.300000 0.400000 0.400000\n", encoding="utf-8")

        import_review(cfg, manifest)

        assert rec.boxes[0].cls == "lemon", "只改类别的修正被丢弃了"
        assert not rec.needs_review

    def test_empty_label_clears_review_flag(self, tmp_path):
        from cvc.config import load_config

        recipe = tmp_path / "r.yaml"
        recipe.write_text(
            f"workspace: {tmp_path / 'ws'}\nlabel:\n  categories: [apple]\n",
            encoding="utf-8",
        )
        cfg = load_config(recipe)
        cfg.ensure_dirs()

        rec = Record(id="000002", image="images/000002.jpg", width=100, height=100, status="labeled")
        rec.boxes = []          # 模型没框到，且被打了待复核
        rec.flag("needs_review")
        rec.flag("no_target")

        manifest = Manifest(cfg.manifest_path)
        manifest.add(rec)
        manifest.save()

        lbl_dir = cfg.review_dir / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (lbl_dir / "000002.txt").write_text("", encoding="utf-8")  # 复核员确认：确实没有目标

        import_review(cfg, manifest)

        assert not rec.needs_review, "复核员确认无目标后，needs_review 应该被清掉"


# ---------------------------------------------------------------- 坐标层


class TestNorm1Tolerance:
    """[m27] 轻微越界的归一化输出被误判成绝对像素。"""

    def test_slightly_over_one_is_norm1(self):
        # 模型常见的越界输出；旧代码会判成 absolute，框塌成原点附近几个像素
        assert detect_format([0.0, 0.0, 1.02, 1.0], 1024, 1024) == "norm1"

    def test_to_pixels_clamps_overflow(self):
        from cvc.utils.bbox_convert import clamp_box

        px = to_pixels([0.0, 0.0, 1.02, 1.0], 1000, 1000, "norm1")
        assert px[2] == pytest.approx(1020.0)
        clipped = clamp_box(px, 1000, 1000)
        assert clipped == (0.0, 0.0, 1000.0, 1000.0)


# ---------------------------------------------------------------- 数据契约


class TestManifestDuplicateIds:
    def test_duplicate_id_rejected_on_load(self, tmp_path):
        p = tmp_path / "m.jsonl"
        rec = {"id": "000001", "image": "a.jpg", "width": 10, "height": 10, "boxes": []}
        p.write_text(
            json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="id 重复"):
            Manifest(p).load()


# ---------------------------------------------------------------- 日志层


class TestLoggingFormatter:
    def test_no_ansi_when_disabled(self, monkeypatch):
        import logging

        from cvc.utils import logging as cvc_logging

        monkeypatch.setattr(cvc_logging, "_use_color", lambda: False)
        rec = logging.LogRecord("cvc", logging.INFO, __file__, 1, "hello", None, None)
        out = cvc_logging._Formatter().format(rec)
        assert "\033[" not in out, "重定向输出时不应带 ANSI 转义码"

    def test_exception_traceback_preserved(self, monkeypatch):
        import logging

        from cvc.utils import logging as cvc_logging

        monkeypatch.setattr(cvc_logging, "_use_color", lambda: False)
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            rec = logging.LogRecord(
                "cvc", logging.ERROR, __file__, 1, "failed", None, sys.exc_info()
            )
        out = cvc_logging._Formatter().format(rec)
        assert "boom" in out and "Traceback" in out, "异常堆栈被吞掉了"


# ---------------------------------------------------------------- 命令层


class TestInitForce:
    """H: `--force` 在文件已存在时是空操作。"""

    def test_force_overwrites_recipe(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from cvc.cli import app

        target = tmp_path / "recipes" / "fruit_picking.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("project: stale\nlabel:\n  categories: [stale]\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, ["init", "-r", str(target), "--force"])
        assert result.exit_code == 0, result.output

        text = target.read_text(encoding="utf-8")
        assert "stale" not in text, "--force 没有真正覆盖 recipe"
        assert "categories" in text


class TestCleanGuards:
    """`cvc clean` 会 rmtree workspace，必须挡住手误。

    这组用例守的是一个**修复过程中引入的二次 bug**：最初的守卫用
    "目录里有没有 images/ 或 manifest.jsonl" 判断是不是 workspace，
    但 `_load()` 里的 `ensure_dirs()` 会先把这些目录创建出来，
    于是 `workspace: .` 顺利通过检查，把整个项目删掉了。
    """

    def _run(self, tmp_path, workspace: str, monkeypatch):
        from typer.testing import CliRunner

        from cvc.cli import app

        proj = tmp_path / "proj"
        proj.mkdir(exist_ok=True)
        recipe = proj / "r.yaml"
        recipe.write_text(f"workspace: {workspace}\nlabel:\n  categories: [a]\n", encoding="utf-8")
        monkeypatch.chdir(proj)
        return CliRunner().invoke(app, ["clean", "-r", str(recipe), "-y"])

    def test_dot_workspace_refused(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        result = self._run(tmp_path, ".", monkeypatch)
        assert result.exit_code != 0
        assert proj.exists(), "workspace: . 把项目目录删掉了"
        assert "拒绝执行" in result.output

    def test_parent_workspace_refused(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, "..", monkeypatch)
        assert result.exit_code != 0
        assert tmp_path.exists()

    def test_root_workspace_refused(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, "/", monkeypatch)
        assert result.exit_code != 0
        assert "拒绝执行" in result.output

    def test_directory_without_manifest_refused(self, tmp_path, monkeypatch):
        result = self._run(tmp_path, "not_a_workspace", monkeypatch)
        assert result.exit_code != 0
        assert "manifest.jsonl" in result.output

    def test_legitimate_workspace_is_deleted(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        ws = proj / "ws"
        ws.mkdir(parents=True)
        (ws / "manifest.jsonl").write_text("", encoding="utf-8")

        result = self._run(tmp_path, "ws", monkeypatch)
        assert result.exit_code == 0, result.output
        assert not ws.exists()
