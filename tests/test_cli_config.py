"""CLI 模板定位与配置加载的测试。

这里守的是一个只在 **非 editable 安装**（pip install 的 wheel）下才会犯的错：
`cvc init` 找不到内置 recipe 模板，退化成几行的最小默认值。
打包配置（pyproject 的 force-include）和查找逻辑要一起对，所以两边都测。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cvc.cli import _find_template
from cvc.config import ConfigError, load_config


class TestFindTemplate:
    def test_template_is_found(self):
        """无论源码安装还是 wheel 安装，都必须能定位到模板。"""
        path = _find_template()
        assert path is not None, "找不到 recipe 模板：wheel 里的 cvc/_recipes/ 可能没打包"
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_template_has_categories(self):
        """模板必须包含关键段落；退化版的 6 行最小配置会在这里挂掉。"""
        path = _find_template()
        assert path is not None
        text = path.read_text(encoding="utf-8")

        for key in ("categories", "distractors", "bbox_format", "workspace", "export", "train"):
            assert key in text, f"模板缺少关键字段：{key}"
        assert len(text.splitlines()) > 50, "模板过短，可能是退化后的最小默认值"

    def test_missing_name_returns_none(self):
        assert _find_template("definitely_not_a_recipe.yaml") is None


class TestLoadConfig:
    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "r.yaml"
        p.write_text(body, encoding="utf-8")
        return p

    def test_minimal_valid(self, tmp_path):
        p = self._write(
            tmp_path,
            "project: t\nlabel:\n  categories: [apple, lemon]\n",
        )
        cfg = load_config(p)
        assert cfg.class_names == ["apple", "lemon"]
        assert cfg.workspace == "workspace"

    def test_empty_categories_rejected(self, tmp_path):
        p = self._write(tmp_path, "project: t\nlabel:\n  categories: []\n")
        with pytest.raises(ConfigError, match="categories"):
            load_config(p)

    def test_duplicate_categories_rejected(self, tmp_path):
        p = self._write(tmp_path, "label:\n  categories: [apple, apple]\n")
        with pytest.raises(ConfigError, match="重复"):
            load_config(p)

    def test_bad_bbox_format_rejected(self, tmp_path):
        p = self._write(
            tmp_path,
            "label:\n  categories: [a]\n  bbox_format: normalized\n",
        )
        with pytest.raises(ConfigError, match="bbox_format"):
            load_config(p)

    def test_bad_backend_rejected(self, tmp_path):
        p = self._write(
            tmp_path,
            "label:\n  categories: [a]\n  backend: gpt\n",
        )
        with pytest.raises(ConfigError, match="backend"):
            load_config(p)

    def test_unknown_field_rejected(self, tmp_path):
        p = self._write(
            tmp_path,
            "label:\n  categories: [a]\ntypo_field: 1\n",
        )
        with pytest.raises(ConfigError, match="未知字段"):
            load_config(p)

    def test_split_must_sum_to_one(self, tmp_path):
        p = self._write(
            tmp_path,
            "label:\n  categories: [a]\nexport:\n  split: {train: 0.5, val: 0.2}\n",
        )
        with pytest.raises(ConfigError, match="1.0"):
            load_config(p)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="找不到"):
            load_config(tmp_path / "nope.yaml")

    def test_paths_are_derived(self, tmp_path):
        p = self._write(tmp_path, "workspace: out\nlabel:\n  categories: [a]\n")
        cfg = load_config(p)
        assert cfg.manifest_path.name == "manifest.jsonl"
        assert cfg.dataset_dir.name == "dataset"
        assert cfg.images_dir.name == "images"
