# docparse — 本地多模态文档解析

基于 Xiaomi-OCR-0（0.8B）+ Ollama 的本地文档解析流水线。设计目标：**尽量不依赖人工复核**。

## 为什么不是「PDF 转图片全丢给模型」

| 输入 | 处理方式 | 原因 |
|---|---|---|
| 矢量 PDF | PyMuPDF 直接抽文字 + bbox + 表格线 | 文本层比任何 OCR 都准，且快 100 倍 |
| 扫描 PDF / 图片 | 渲染 → VLM | 无文本层或文本层损坏 |
| 混合页 | 逐页判断 | 文本层质量分决定走哪条路 |
| docx/pptx/xlsx | 原生 XML 解析（后续） | 不需要 OCR |

分诊在**页级**进行，不是文档级——真实 PDF 几乎都是混合的。

## 自动质量门（替代人工复核）

```
page OCR → 重复检测 ──失败──→ presence_penalty=1.5 重试
            │                      │
            ├──截断/空输出──────────┤
            ▼                      ▼
        通过                    仍失败 → region 模式（版面检测+逐块）
                                   │
                                   ▼
                              仍失败 → JSON 中 needs_review=true + 原因
```

- **重复/循环**：复用上游 `repetition_guard`，自动重试一次
- **截断**：`finish_reason=length` 直接判定失败
- **表格**：OTSL → HTML，转换失败保留原始 OTSL 并告警
- **KIE 幻觉**：抽取的每个字段回查 OCR 正文，数字按位比对（`1,234.56` 与 `1234.56` 视为一致），查不到即标记 `field_not_grounded`
- 所有告警带 `severity`，只有 `high` 才会把整篇标为 `needs_review`

## 输出

- `output/<name>.md` — Markdown（OTSL 表格已转 HTML，公式为 LaTeX）
- `output/<name>.json` — 完整 IR：每页/每块含 `bbox`、`type`、`source`、`confidence`、`warnings`、`kie`

Markdown 只是 IR 的一个视图；后续做 RAG、高亮、增量重跑都从 JSON 出发。

## 运行

```bash
# 0. 环境（一次性）
conda create -y -n docparse -c conda-forge python=3.11
conda activate docparse
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r docparse/requirements.txt

# 1. 启动 Ollama（16384 上下文，8GB 显卡专用参数）
docparse/scripts/start_ollama.sh

# 2. 自检
python -m docparse doctor

# 3. 解析
python -m docparse parse 文件.pdf 图片.jpg -o output
python -m docparse parse 发票.jpg --fields "发票号码,开票日期,价税合计"
```

> 工作目录：`/home/wonder/work/xiaomiocr`，`docparse` 与官方仓库 `Xiaomi-OCR-0` 平级。

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `auto` | `auto`=整页优先，失败升级到 region；`region`=强制版面分块；`page`=强制整页 |
| `--dpi` | 200 | 扫描页渲染 DPI。整页模式超过约 176 DPI 会被模型侧压到 2048 长边；region 模式裁块才能吃到高 DPI 的收益 |
| `--max-pages` | 全部 | 调试用 |
| `--fields` | 无 | 逗号分隔的 KIE 字段 |
| `--no-retry` | 关 | 关闭自动重复重试 |

## 本机适配

- GPU：RTX 4060 Laptop 8GB —— VLM 独占；版面模型强制 CPU（`layout_device="cpu"`）
- Ollama 默认上下文只有 4096，整页 2048px 图像本身就要 2–4k vision token，**必须设 `OLLAMA_CONTEXT_LENGTH=16384`**，启动脚本已处理
- `OLLAMA_NUM_PARALLEL=1`：低并发场景下避免显存翻倍

## 已知边界

- Ollama 后端在上游属于 **experimental**，官方只在 Apple Silicon 做过一致性抽检；基准分数不等于 Ollama 分数
- 手写、书法、古籍：用 `--mode page`（region 切块会丢上下文）
- 表格跨页合并、合并单元格的 span 还原尚未实现
- `region` 模式需要 PaddleX + PP-DocLayoutV3（首次使用自动下载）
