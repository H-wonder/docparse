# docparse

本地多模态文档解析系统：**Xiaomi-OCR-0 (0.8B) + Ollama**，面向扫描件、发票/票据、合同、论文、Office 文档。
核心设计目标是**尽量不依赖人工复核**——所有可检测的失败模式都走自动重试/升级，最终以结构化告警输出。

## 目录结构

```
.
├── docparse/                 # 本项目代码
│   ├── ir.py                 # 文档中间表示（Markdown/JSON 只是它的视图）
│   ├── pipeline.py           # 分诊 + OCR + 自动升级阶梯
│   ├── pdfio.py              # PDF 文本层探测、原生抽取、渲染
│   ├── backend.py            # Ollama OpenAI 兼容客户端
│   ├── quality.py            # 自动质量门（重复/截断/覆盖率/KIE 幻觉）
│   ├── layout.py             # PP-DocLayoutV3（CPU）
│   ├── office.py             # docx/pptx/xlsx/txt 原生解析
│   ├── writers.py            # IR → Markdown/JSON
│   ├── evaluate.py           # 与参考 Markdown 对比（CER/相似度）
│   ├── cli.py
│   ├── requirements.txt
│   ├── scripts/start_ollama.sh
│   └── README.md             # 设计细节与参数说明
└── Xiaomi-OCR-0/             # 上游仓库（gitignore，需自行拉取）
```

## 依赖的上游仓库（不纳入本仓库）

上游仓库包含模型/示例/评测数据，体积大，不适合同步。请克隆到与本仓库**同级**的目录：

```bash
git clone --depth 1 https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
```

`docparse` 会从 `../Xiaomi-OCR-0` 导入它的 `postprocess`（OTSL 表格转换、重复检测、Markdown 合并）和 `evaluation_pipeline`（裁块）。

## 环境

```bash
conda create -y -n docparse -c conda-forge python=3.11
conda activate docparse
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r docparse/requirements.txt
```

## 模型运行时

Ollama **必须 ≥ 0.35**（模型架构为 `qwen35`；0.20.x 会报 `unknown model architecture`）。

```bash
# 启动（自动设置 16384 上下文、单并发；模型不存在会先拉取）
docparse/scripts/start_ollama.sh
```

`start_ollama.sh` 的关键参数：

| 变量 | 值 | 原因 |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | 16384 | 默认 4096 装不下整页图像 + 输出 |
| `OLLAMA_NUM_PARALLEL` | 1 | 8GB 显存下避免 KV 翻倍 |
| `OLLAMA_MODEL` | `longwayxu/xiaomi-ocr-0:bf16` | 社区 BF16 GGUF（含视觉投影器，约 1.8GB） |

## 使用

```bash
python -m docparse doctor

# 通用文档
python -m docparse parse 文件.pdf 扫描件.jpg -o output

# 票据 KIE（字段级幻觉回查）
python -m docparse parse 发票.jpg --fields "发票号码,开票日期,价税合计"

# 在你的数据上评测（A/B mode、DPI、后端版本）
python -m docparse eval samples/ --refs refs/ --mode auto -o eval_report.json
```

输出：`output/<name>.md` 和 `output/<name>.json`（IR，含每块 bbox、source、confidence、warnings）。

架构、自动质量门、参数含义见 [docparse/README.md](docparse/README.md)。

## 已知边界

- Ollama 后端在上游属于 experimental，官方只在 Apple Silicon 做过一致性抽检
- 手写/书法/古籍建议 `--mode page`；表格密集的印刷件建议 `--mode region`
- 跨页表格合并、合并单元格 span 还原尚未实现
- `region` 模式需要 PaddleX + PP-DocLayoutV3（首次使用自动下载）
