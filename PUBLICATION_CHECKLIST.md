# GitHub 发布检查清单

本文件记录当前发布版文件夹的检查结果。

发布版路径：

```text
<local-release-folder>
```

## 1. API key 泄露检查

状态：通过。

检查结果：

- 未发现真实 API key。
- `.env.example` 中只保留占位符：

```text
LLM_API_KEY="your-api-key"
```

注意：

- 不要把自己的 `.env` 上传到 GitHub。
- 不要把 DeepSeek、OpenAI、Tavily、SerpApi 等真实密钥写进 README 或代码。

## 2. .env 是否被上传

状态：通过。

发布版中只包含：

```text
.env.example
```

不包含：

```text
.env
```

`.gitignore` 已明确忽略 `.env` 和 `.env.*`，但保留 `.env.example`。

## 3. 无用缓存文件检查

状态：通过。

未发现：

```text
__pycache__/
.pytest_cache/
*.pyc
*.log
```

当前发布版没有运行日志和 Python 缓存。

## 4. 数据文件公开适配

状态：通过，已做精简。

包含的数据：

```text
data/demo_cases.json
data/sample_reviews.csv
data/knowledge/
data/eval/customer_service_eval_100.csv
data/eval/customer_service_eval_100_results.csv
data/eval/customer_service_eval_100_errors.csv
data/eval/customer_service_eval_100_summary.json
data/jddc/README.md
data/jddc/sample_jddc_dialogues.jsonl
```

未包含的数据：

```text
data/raw/
data/labeled/
大模型权重
本地 tokenizer
真实私有数据
```

说明：

- `data/eval` 用于展示项目评估流程和复现实验指标。
- `data/knowledge` 是本地 RAG 售后政策和客服话术示例。
- `data/jddc` 只保留样例和说明，不包含原始 JDDC 数据集。

## 5. README 复现性检查

状态：通过。

README 已包含：

- 项目简介
- 系统架构
- 技术栈
- 快速开始
- `.env.example` 配置说明
- 命令行运行方式
- Streamlit 运行方式
- 测试命令
- 评估指标
- 项目总结
- 后续路线图

本地绝对路径已经改为占位路径，例如：

```text
external/LALUN/delivery_105
models/hfl_chinese_macbert_base
```

## 6. LALUN 相关版权和路径问题

状态：通过，已做隔离。

发布版没有包含：

- LALUN 原始仓库代码
- LALUN 模型权重
- 中文 MacBERT 模型文件
- 本地训练 checkpoint

只包含：

- LALUN 适配器
- LALUN 推理/微调辅助脚本
- LALUN 接入说明文档

这些内容用于说明如何对接本地 LALUN，不直接分发第三方模型或论文代码。

## 7. .gitignore 检查

状态：通过。

已新增：

```text
.gitignore
```

主要忽略：

- `.env`
- 缓存文件
- 日志文件
- 虚拟环境
- IDE 文件
- 原始数据
- 标注数据
- 模型权重
- checkpoint
- Streamlit secrets

## 8. Git 初始化和 GitHub 上传

当前状态：

- 发布版文件夹已经准备好。
- 自动化测试通过：`27 passed`。
- 还需要根据本机是否安装 Git 来初始化仓库。

推荐命令：

```powershell
cd <local-release-folder>
git init
git add .
git commit -m "Initial release: ecommerce after-sales agent"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

推荐仓库名：

```text
ecommerce-after-sales-agent
```

或：

```text
langgraph-rag-customer-service-agent
```

## 当前发布版摘要

```text
文件数：83
总大小：约 461 KB
测试结果：27 passed
真实密钥：未发现
.env：未包含
缓存/日志：未发现
大数据/模型权重：未包含
```
