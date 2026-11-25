"""阶段四 · 可视化质检 + 人工复核（HITL）闭环。

自动标注不可能 100% 正确，所以这一阶段做两件事：

**A. 可视化**：给每张图画框，生成一份可直接用浏览器打开的 HTML 报告。
   按 `needs_review` / `no_target` / `label_failed` 分组，优先看最可疑的。

**B. 人工复核闭环**：
   `export_review()`  把待复核样本导出成 YOLO txt（任何 YOLO 标注工具都能打开）
   `import_review()`  把改好的 txt 读回来，覆盖 manifest 里的框，并标记 source=human

这样"AI 先标 80%、人只改 20%"的收益才真正落地。
"""

from __future__ import annotations

import html
from pathlib import Path

from ..config import Config
from ..utils.logging import banner, get_logger
from ..utils.manifest import Box, Manifest

log = get_logger("visualize")

# 类别固定配色，保证多次生成颜色一致
PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (255, 165, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 215, 0),
    (199, 21, 133),
]


def _color_for(cls: str, categories: list[str]) -> tuple[int, int, int]:
    idx = categories.index(cls) if cls in categories else 0
    return PALETTE[idx % len(PALETTE)]


def draw_overlay(cfg: Config, rec, categories: list[str]) -> Path:
    """在原图上画框 + 类名，输出到 overlays/。"""
    import cv2

    src = cfg.workspace_path / rec.image
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(src)

    for b in rec.boxes:
        color = _color_for(b.cls, categories)
        p1 = (int(b.x1), int(b.y1))
        p2 = (int(b.x2), int(b.y2))
        cv2.rectangle(img, p1, p2, color, 2)

        tag = f"{b.cls} {b.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(
            img, tag, (p1[0] + 2, max(12, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    if rec.needs_review:
        cv2.putText(
            img, "NEEDS REVIEW", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )

    out = cfg.overlays_dir / f"{rec.id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def _card(rec, rel_overlay: str, rel_thumb: str) -> str:
    flags = " ".join(f'<span class="flag">{html.escape(f)}</span>' for f in rec.flags)
    cls_count: dict[str, int] = {}
    for b in rec.boxes:
        cls_count[b.cls] = cls_count.get(b.cls, 0) + 1
    detail = " / ".join(f"{k}×{v}" for k, v in cls_count.items()) or "无目标"
    return f"""
    <figure class="card">
      <a href="{rel_overlay}" target="_blank"><img src="{rel_thumb}" loading="lazy" alt="{rec.id}"></a>
      <figcaption>
        <b>{rec.id}</b> · {detail}
        <div class="flags">{flags}</div>
      </figcaption>
    </figure>"""


_HTML_HEAD = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 24px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
  .stat {{ background: #fff; border: 1px solid #e3e3e3; border-radius: 10px;
           padding: 10px 16px; min-width: 96px; }}
  .stat b {{ display: block; font-size: 20px; }}
  .stat span {{ color: #777; font-size: 12px; }}
  h2 {{ font-size: 15px; margin: 28px 0 12px; padding-left: 8px;
        border-left: 3px solid #3b82f6; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
  .card {{ margin: 0; background: #fff; border: 1px solid #e3e3e3; border-radius: 10px;
           overflow: hidden; }}
  .card img {{ width: 100%; display: block; aspect-ratio: 4/3; object-fit: cover; }}
  figcaption {{ padding: 8px 10px; font-size: 12px; color: #444; }}
  .flags {{ margin-top: 4px; }}
  .flag {{ display: inline-block; background: #fee2e2; color: #b91c1c;
           border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-right: 4px; }}
  .empty {{ color: #888; font-size: 13px; }}
</style></head><body>
<h1>{title}</h1>
<div class="meta">生成时间 {ts} · 模型 {model} · 类别 {cats}</div>
"""


def build_html_report(cfg: Config, manifest: Manifest) -> Path:
    """生成自包含索引的 HTML 质检报告（图片走相对路径，不内嵌 base64）。"""
    from datetime import datetime

    s = manifest.summary()
    out_path = cfg.workspace_path / "report.html"

    groups: dict[str, list] = {
        "需人工复核": [],
        "模型判定无目标（负样本）": [],
        "标注失败": [],
        "正常": [],
    }

    for rec in manifest.records:
        if not rec.is_usable:
            continue
        if "label_failed" in rec.flags:
            groups["标注失败"].append(rec)
        elif rec.needs_review:
            groups["需人工复核"].append(rec)
        elif not rec.has_target:
            groups["模型判定无目标（负样本）"].append(rec)
        else:
            groups["正常"].append(rec)

    parts = [
        _HTML_HEAD.format(
            title=f"{cfg.project} · 数据集质检报告",
            ts=datetime.now().strftime("%Y-%m-%d %H:%M"),
            model=html.escape(cfg.label.model),
            cats=html.escape(", ".join(cfg.class_names)),
        ),
        '<div class="stats">',
    ]
    for label, value in (
        ("总样本", s["total"]),
        ("可用", s["usable"]),
        ("已剔除", s["rejected"]),
        ("正样本", s["positives"]),
        ("负样本", s["negatives"]),
        ("待复核", s["needs_review"]),
    ):
        parts.append(f'<div class="stat"><b>{value}</b><span>{label}</span></div>')
    parts.append("</div>")

    if s["boxes_per_class"]:
        parts.append('<div class="stats">')
        for cls, n in sorted(s["boxes_per_class"].items()):
            parts.append(f'<div class="stat"><b>{n}</b><span>{html.escape(cls)} 框数</span></div>')
        parts.append("</div>")

    for title, recs in groups.items():
        parts.append(f"<h2>{html.escape(title)} · {len(recs)}</h2>")
        if not recs:
            parts.append('<p class="empty">（无）</p>')
            continue
        parts.append('<div class="grid">')
        for rec in recs[:400]:  # 单页上限，避免浏览器卡死
            overlay = cfg.overlays_dir / f"{rec.id}.jpg"
            thumb_rel = f"overlays/{rec.id}.jpg" if overlay.exists() else f"images/{Path(rec.image).name}"
            parts.append(_card(rec, f"overlays/{rec.id}.jpg", thumb_rel))
        parts.append("</div>")
        if len(recs) > 400:
            parts.append(f'<p class="empty">仅显示前 400 张，共 {len(recs)} 张。</p>')

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def run(cfg: Config, manifest: Manifest) -> dict:
    """画框 + 生成 HTML 报告。"""
    banner("阶段四 · 可视化质检")
    cfg.ensure_dirs()

    targets = [r for r in manifest.usable() if r.status in ("labeled", "exported")]
    if not targets:
        log.info("没有可可视化的记录（先跑 label）")
        return {"drawn": 0}

    drawn = 0
    for rec in targets:
        try:
            draw_overlay(cfg, rec, cfg.class_names)
            drawn += 1
        except Exception as exc:
            log.warning(f"{rec.id} 画框失败：{exc}")

    report = build_html_report(cfg, manifest)
    log.info(f"已生成 {drawn} 张叠加图 → {cfg.overlays_dir}")
    log.info(f"质检报告 → {report}")
    return {"drawn": drawn, "report": str(report)}


# ---------------------------------------------------------------- HITL 复核


def export_review(cfg: Config, manifest: Manifest, only_flagged: bool = True) -> Path:
    """导出待复核样本为 YOLO txt，供 LabelImg / X-AnyLabeling 等工具修正。

    目录结构：
        workspace/review/
        ├── images/000012.jpg
        ├── labels/000012.txt      # YOLO 格式，直接改
        └── README.txt             # 给同学看的操作说明
    """
    banner("阶段四 · 导出复核包")

    pool = [
        r for r in manifest.usable()
        if r.status in ("labeled", "exported") and (r.needs_review or not only_flagged)
    ]
    if not pool:
        log.info("没有需要复核的样本 👍")
        return cfg.review_dir

    img_dir = cfg.review_dir / "images"
    lbl_dir = cfg.review_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    names = cfg.class_names
    for rec in pool:
        src = cfg.workspace_path / rec.image
        dst = img_dir / f"{rec.id}.jpg"
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

        lines = []
        for b in rec.boxes:
            if b.cls not in names:
                continue
            cid = names.index(b.cls)
            cx, cy, w, h = b.to_yolo(rec.width, rec.height)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{rec.id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    readme = f"""复核包使用说明
================

本目录包含 {len(pool)} 张待复核图片。

图片：review/images/<id>.jpg
标签：review/labels/<id>.txt   （YOLO 格式：class_id cx cy w h，全部归一化 0~1）

类别编号：
{chr(10).join(f'  {i} = {n}' for i, n in enumerate(names))}

操作步骤：
1. 用任意支持 YOLO 格式的标注工具打开 review/images 目录
   （LabelImg / X-AnyLabeling / Label Studio 均可）
2. 修正错误框：删掉误检、补上漏检、调整位置
3. 保存后回到命令行执行：
       cvc review-import -r recipes/fruit_picking.yaml
4. 人工修改过的框会被标记为 source=human，并解除 needs_review

注意：
- 如果某张图其实没有目标水果，把对应 txt 清空即可（作为负样本很有价值）
- 不要改文件名，id 是回写 manifest 的唯一凭据
"""
    (cfg.review_dir / "README.txt").write_text(readme, encoding="utf-8")

    log.info(f"已导出 {len(pool)} 个样本 → {cfg.review_dir}")
    log.info("用标注工具改完后执行：cvc review-import")
    return cfg.review_dir


def import_review(cfg: Config, manifest: Manifest) -> dict:
    """把复核后的 YOLO txt 读回 manifest。"""
    banner("阶段四 · 回写人工复核结果")

    lbl_dir = cfg.review_dir / "labels"
    if not lbl_dir.exists():
        log.warning(f"找不到复核标签目录：{lbl_dir}（先执行 cvc review-export）")
        return {"updated": 0}

    names = cfg.class_names
    updated = cleared = 0

    for txt in sorted(lbl_dir.glob("*.txt")):
        rec = manifest.get(txt.stem)
        if rec is None:
            log.debug(f"manifest 中没有 id={txt.stem}，跳过")
            continue

        boxes: list[Box] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                boxes.append(Box.from_yolo(line, rec.width, rec.height, names))
            except (ValueError, IndexError) as exc:
                log.warning(f"{txt.name} 有一行解析失败：{line!r}（{exc}）")

        for b in boxes:
            b.source = "human"

        if len(boxes) == len(rec.boxes) and all(
            abs(a.x1 - b.x1) < 0.5 and abs(a.y1 - b.y1) < 0.5
            and abs(a.x2 - b.x2) < 0.5 and abs(a.y2 - b.y2) < 0.5
            for a, b in zip(rec.boxes, boxes, strict=False)
        ):
            continue  # 没改过，不重复计数

        rec.boxes = boxes
        rec.unflag("needs_review")
        rec.unflag("label_failed")
        if boxes:
            rec.unflag("no_target")
        else:
            rec.flag("no_target")
        rec.meta["reviewed"] = True

        if boxes:
            updated += 1
        else:
            cleared += 1

    manifest.save()
    log.info(f"已更新 {updated} 张（有人工框），清空为负样本 {cleared} 张")
    return {"updated": updated, "cleared": cleared}
