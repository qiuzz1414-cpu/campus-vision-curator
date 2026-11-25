# campus-vision-curator (cvc)

**从一堆乱拍的照片，到一份能直接训练的干净 YOLO 数据集。**

面向大学生课程作业 / 学科竞赛的轻量级视觉数据策展流水线。一条 YAML 配置驱动六个阶段，用 VLM 自动打框 + 人工只改错的那部分。

> 灵感来自 NVIDIA Cosmos Curator 的"配置驱动 + 阶段解耦 + 数据闭环"思想，但把 Ray 集群、S3、Lance 全部换成本地文件系统 + JSONL，一台笔记本就能跑。

---

## 为什么需要它

做 CV 课程作业时，真正耗时的不是调模型，而是：

- 手机拍的竖图导入后全是横的（EXIF 旋转）
- 连拍 200 张，其中 150 张几乎一样
- 手动框 500 个水果，框到怀疑人生
- 训练集里某个类别一张都没有，训完发现 0 召回
- 报告里要"数据集统计表"，只能手数

这六个坑，正好对应本项目的六个阶段。

---

## 架构

```
                     recipe.yaml  （唯一的配置入口）
                          │
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│   s1   │→ │   s2   │→ │   s3   │→ │   s4   │→ │   s5   │→ │   s6   │
│ ingest │  │ dedup  │  │ label  │  │ review │  │ export │  │ train  │
└────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
 照片/视频   模糊+去重   VLM打框    可视化/     YOLO格式    训练+
 EXIF修正   pHash/CLIP  OpenAI兼容  人工修正    划分数据集  指标汇总
    │           │           │           │           │           │
    └───────────┴───────────┴───────────┴───────────┴───────────┘
                          ▼
                   manifest.jsonl
              （唯一的中间数据契约）
```

**核心设计：manifest 是唯一真相源。**

`manifest.jsonl` 是一行一条 JSON 的文件，记录每张图的路径、尺寸、哈希、清晰度、标注框、复核状态。每个阶段只做「读 manifest → 改自己那部分字段 → 写回」。

因此：

| 好处 | 具体表现 |
|---|---|
| 可断点续跑 | 中断后重跑，已完成的自动跳过 |
| 可插拔 | 想换掉 VLM？只改 s3，其他五阶段不受影响 |
| 可审计 | 任何一张图为什么被剔除、哪一步改过，全在 manifest 里 |
| 可对比 | 人工改了哪些框，和 AI 原来的差多少，能直接 diff |

---

## 快速开始

```bash
git clone <your-repo-url> campus-vision-curator
cd campus-vision-curator

# 推荐 uv；也可以 pip install -e .
uv pip install -e ".[train]"

# 1. 初始化工作区
cvc init -r recipes/fruit_picking.yaml

# 2. 自检：依赖、图片目录、VLM 服务连通性
cvc doctor -r recipes/fruit_picking.yaml

# 3. 把照片丢进 raw/
mkdir -p raw && cp ~/Pictures/apples/*.jpg raw/

# 4. 一键跑完（或按阶段分步跑）
cvc run -r recipes/fruit_picking.yaml
```

产物：

```
workspace/
├── manifest.jsonl           # 数据契约
├── recipe.snapshot.yaml     # 本次运行的配置快照（可复现）
├── images/                  # 规范化后的图
├── overlays/                # 画了框的预览图
├── rejects/                 # 被剔除的图（保留证据）
├── report.html              # ★ 浏览器打开，质检报告
├── review/                  # ★ 人工复核包
└── dataset/
    ├── data.yaml            # 直接喂给 ultralytics
    ├── stats.md             # ★ 数据集统计表，可直接贴进报告
    └── images/ labels/
```

带 ★ 的三样，是交作业时最省事的东西。

---

## 分阶段使用

每个命令都是幂等的，可以反复跑。

```bash
cvc ingest          # 照片/视频 → 统一尺寸、修 EXIF 旋转
cvc dedup           # 剔模糊 + pHash 去重（可选 CLIP 语义去重）
cvc label           # VLM 自动标注
cvc visualize       # 生成 report.html 质检报告
cvc review-export   # 导出 review/ 给标注工具
cvc review-import   # 人工改完回写 manifest
cvc export          # 导出 YOLO 数据集 + stats.md
cvc train           # 训练 + 指标汇总
cvc status          # 看当前进度
```

### 人工复核闭环（HITL）

自动标注的定位是「省掉 80% 的点击」，不是「消灭人工」。

```bash
cvc review-export                       # 只导出 needs_review 的样本
# 用 LabelImg / X-AnyLabeling 打开 workspace/review/images/，改框
cvc review-import                       # 回写，人工框标记为 source=human
```

工具无关——只要能读写 YOLO txt 就行。

---

## 配置说明（recipe.yaml）

只改 YAML，不改代码。关键字段：

### 目标类别与干扰物

```yaml
label:
  categories: [apple, lemon]        # ★ 只标这两类
  distractors: [pear, orange, guava, leaf, hand]   # ★ 必须忽略
```

`distractors` 是效果最明显的配置项。VLM 默认很热情，会把画面里所有水果都框上；显式告诉它"这些看见了也不许标"，误检率能降一大截。

### 坐标格式

```yaml
label:
  bbox_format: auto    # auto | absolute | norm1000 | norm1
```

| 模型 | 建议值 |
|---|---|
| Qwen2.5-VL / GPT-4o | `absolute` |
| Qwen3-VL | `norm1000` |
| LLaVA / InternVL 部分版本 | `norm1` |
| 不确定 | `auto`（日志会打印每张图实际猜到的格式） |

生产环境建议写死。`auto` 用的是启发式判断，个别图会猜错，表现是"框整体偏移"。

### 推理后端

任何 OpenAI 兼容端点都能用：

```yaml
label:
  # 本地 vLLM
  base_url: http://127.0.0.1:8000/v1
  api_key: EMPTY
  model: Qwen2.5-VL-7B-Instruct

  # 或者云 API（没显卡的同学）
  # base_url: https://api.siliconflow.cn/v1
  # api_key: sk-xxxx
```

本地起服务：

```bash
bash scripts/serve_vllm.sh                                  # 默认 7B，约 16GB 显存
bash scripts/serve_vllm.sh Qwen/Qwen2.5-VL-3B-Instruct      # 显存不够降级
```

---

## 实战案例：苹果 + 柠檬采摘机器人

`recipes/fruit_picking.yaml` 就是从一个真实课程作业沉淀下来的配置。作业要求：

- 只能识别本组指定的两种水果（苹果 + 柠檬）
- 验收时机器上会挂其他水果做干扰，摘错了就算失败
- 需要交一份含数据集统计和训练指标的报告

对应配置：

```yaml
label:
  categories: [apple, lemon]
  distractors: [pear, orange, guava, tomato, leaf, hand]
```

`pear` 和 `guava` 不是随便写的——它们跟柠檬颜色形状接近，是最容易误检的干扰物，必须在采集阶段就放进背景，让模型学会忽略。

跑完之后 `stats.md` 和 `summary.md` 直接就是报告要的表格。

---

## 六个阶段的技术细节

### s1 ingest

- `ImageOps.exif_transpose()` 修手机照片方向（不修的话竖拍图全是横的，框会整体错位）
- 视频按 `video_fps` 抽帧
- sha256 干掉字节级重复
- 统一转 RGB、限最长边、重编码 JPG

### s2 dedup

三道闸门，从便宜到贵：

| 闸门 | 指标 | 阈值经验 |
|---|---|---|
| 尺寸 | 短边像素 | < 200 弃用 |
| 清晰度 | Laplacian 方差 | 清晰 >100，虚焦 30~100，糊 <30 |
| 重复 | pHash 汉明距离 | ≤6 视为重复 |
| 语义重复（可选） | CLIP 余弦相似度 | ≥0.95 |

被剔除的图移到 `rejects/` 而不是删除——报告里要写"过滤掉多少、为什么"。

### s3 label

- 异步并发，`concurrency` 控制
- 指数退避重试（网络抖动、服务端限流）
- 单张失败不影响整批
- 低置信度不丢弃，打 `needs_review` 进复核队列
- 类别名模糊匹配（`apples` / `Apple` / `a red apple` 都能对上 `apple`）

### s4 visualize / review

- 按「需复核 / 负样本 / 标注失败 / 正常」分组出 HTML
- 复核包导出成 YOLO txt，标注工具无关
- 回写时标记 `source=human`，可与 AI 框对比

### s5 export

三个容易踩的坑，代码里专门处理了：

1. **分层划分**：按主类别分层，保证 train 覆盖所有类别（否则可能某类全在 val，训出来 0 召回）
2. **负样本掺入**：按 `negative_ratio` 把无目标图掺进 train，降低误检
3. **空标签合法**：负样本的 txt 是空文件，不是缺失文件

导出后主动检查类别覆盖，缺了会打警告：

```
⚠️ train 划分中缺少类别：['lemon']，训练结果对应类别将无法识别
```

### s6 train

封装 Ultralytics，训练完自动把 `results.csv` 转成 Markdown 指标表。

ultralytics 是惰性导入——没装训练依赖也能跑完 s1~s5。

---

## 项目结构

```
campus-vision-curator/
├── README.md
├── pyproject.toml
├── recipes/
│   └── fruit_picking.yaml      # 作业 Demo 配置
├── src/cvc/
│   ├── cli.py                  # 命令入口
│   ├── config.py               # recipe 加载与校验
│   ├── stages/
│   │   ├── s1_ingest.py
│   │   ├── s2_dedup.py
│   │   ├── s3_autolabel.py
│   │   ├── s4_visualize.py
│   │   ├── s5_export.py
│   │   └── s6_train.py
│   └── utils/
│       ├── manifest.py         # 数据契约
│       ├── bbox_convert.py     # 坐标归一化
│       ├── vlm.py              # VLM 调用与响应解析
│       └── logging.py
├── scripts/
│   └── serve_vllm.sh
└── tests/
```

---

## 常见问题

**Q：没有显卡能跑吗？**
能。`label` 阶段连云端 OpenAI 兼容 API（硅基流动、百炼、DeepSeek-VL 等），`train` 阶段用 CPU 也能训 yolov8n，只是慢。

**Q：框整体偏移怎么办？**
九成是坐标格式问题。把 `bbox_format` 从 `auto` 改成写死值试试。如果所有框都往一个方向偏，是**发给模型的尺寸和换算用的尺寸不一致**——本项目用 `image_max_side` 缩小后按缩小尺寸换算，已处理，但自定义模型可能返回原图坐标。

**Q：一个框都没标出来？**
看 `report.html` 里「模型判定无目标」那一组。如果图里明明有水果，检查 `categories` 拼写，以及 VLM 是否真的支持视觉输入（部分文本模型会返回"我看不到图片"）。

**Q：类别名应该用中文还是英文？**
英文。YOLO 的 `data.yaml`、标注工具、Ultralytics 全链路对英文最稳。

**Q：review-import 之后框没变？**
检查 `review/labels/<id>.txt` 的文件名是否和 manifest 的 id 一致。id 是回写的唯一凭据，不能改文件名。

---

## License

MIT
