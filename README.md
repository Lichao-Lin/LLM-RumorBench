<div align="center">

# LLM-RumorBench

**From model agreement to paired intervention analysis**

大语言模型谣言识别 · 干预评估 · 可复核的统计实验

[![Tests](https://github.com/Lichao-Lin/LLM-RumorBench/actions/workflows/tests.yml/badge.svg)](https://github.com/Lichao-Lin/LLM-RumorBench/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB)](requirements.txt)
[![Paper](https://img.shields.io/badge/IEEE-Paper-00629B)](https://doi.org/10.1109/CAIBDA65784.2025.11183156)

[快速开始](#快速开始) · [研究扩展](#研究扩展) · [数据规范](docs/protocol.md) · [复现边界](#与论文的关系)

</div>

研究搜索与深度思考两种干预，如何改变大语言模型的谣言识别表现。输入已收集的模型回答与真实标签，在同一批样本上评估四个条件，输出一致性指标、配对差异和不确定性。

## 与论文的关系

**Evaluating the Impact of Multimodal Cognitive and Informational Interventions on Rumor Detection in Large Language Models: A DeepSeek Case Study**

[IEEE / DOI: 10.1109/CAIBDA65784.2025.11183156](https://doi.org/10.1109/CAIBDA65784.2025.11183156)

本仓库保留论文相关的历史脚本、图片与 SPSS 输出，并新增离线评估框架。**原始样本、完整标签字典、提示词、模型快照和采集记录尚未公开在此仓库，当前不能据此完整复现论文表格或结论。** 新增统计扩展不是原论文已报告结果。

仓库不调用 DeepSeek API，也不实现图片/音频推理或搜索工具。论文题目中的 multimodal 不代表本代码已具备多媒体推理能力。当前扩展分析的是四种已采集条件的标签输出。

## 研究扩展

| 层次 | 已实现能力 | 回答的问题 |
| :--- | :--- | :--- |
| 数据检查 | 唯一样本 ID、标签全集、缺失率、共同样本数 | 比较的是否是同一批样本？ |
| 分类与一致性 | Accuracy、固定标签集 Macro-F1、Cohen’s κ、混淆矩阵 | 判断正确多少，与真值一致多少？ |
| 序数分析 | 显式开启的线性/二次加权 κ、Spearman | 错误是否跨越多个等级？ |
| 配对比较 | 相对 Neither 的准确率差、95% bootstrap CI | 改善幅度有多大、估计有多稳定？ |
| 假设检验 | 精确 McNemar、三项基线比较的 Holm 校正 | 改变是否超出配对随机波动？ |
| 因子对比 | 搜索主效应、思考主效应、两者交互对比 | 联合变化是否等于单独变化之和？ |
| 可追溯性 | 数据与代码 SHA-256、随机种子、依赖版本、配置 | 能否用同一输入复算？ |

所有配对比较使用四条件均完整的共同样本；同时保留各条件可用样本指标。交互对比是探索性统计量，不能直接证明因果机制。详见[实验协议与限制](docs/protocol.md)。

## 快速开始

需要 Python 3.11 或 3.12。在独立环境安装固定版本依赖：

```bash
git clone https://github.com/Lichao-Lin/LLM-RumorBench.git
cd LLM-RumorBench
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m rumorbench examples/synthetic.csv --labels 0 1 2 --ordinal --output outputs/demo
python -m unittest discover -s tests -v
```

`examples/synthetic.csv` 是 **12 个手工构造的功能测试样本**，不是论文数据，也不是模型性能证据。这里的 0/1/2 仅演示序数编码。真实任务若是无序类别，去掉 `--ordinal`，并传入真实的标签全集。

输出目录必须是新目录，防止覆盖已有实验：

| 文件 | 内容 |
| :--- | :--- |
| `results.json` | 完整指标、混淆矩阵、交互对比、覆盖率、配置与版本记录 |
| `metrics.csv` | 各条件的可用样本与共同样本指标 |
| `comparisons.csv` | 三个条件相对 Neither 的配对比较 |

输入支持 UTF-8 CSV 和 XLSX（第一张工作表）。必需列：

```text
sample_id,correct answer,Neither,web search,deep think,Both
```

每行是一个独立样本的一次四条件实验。空预测可以保留；真值不能为空；拒绝重复 ID 与未声明标签。只有空白单元格被视为缺失，字符串 `NA` 可以是合法类别。重复试验、同事件多条新闻应使用独立的分层/聚类设计，不能直接混入这里。

## 本地运行 DeepSeek

新增 [本地部署示例](deploy/local-llm/README.md)：用 Docker 启动 Ollama，
以 DeepSeek-R1-Distill-Qwen-1.5B 为例，提供中文注释的 [Python 调用代码](deploy/local-llm/chat.py)。
支持 CPU，另附 NVIDIA GPU 配置；默认只向本机开放端口。部署说明包含启动、下载模型、
请求与停止命令。该示例用于采集模型回答，评分仍使用上面的离线评估流程。

## 文件关系

```text
rumorbench/
  evaluation.py       校验、指标、配对检验与因子对比
  __main__.py         CSV/XLSX → 分析 → JSON/CSV
examples/synthetic.csv 仅用于演示与自动测试
tests/                已知答案、缺失错位、边界输入、CLI 回归测试
docs/protocol.md       实验设计、统计假设、扩展路线
requirements.txt      固定版本运行环境
kappa.py              历史四列 Excel 入口（已修复样本错配）
heatmap coding2.py    历史相关热图入口（参数化，保留负相关）
heatmap.PNG           历史图，不是新增扩展的运行结果
描述性统计.spv        历史 SPSS 输出，不是原始数据
```

旧入口仍可使用：

```bash
python kappa.py path/to/four_columns.xlsx --ordinal-labels 0 1 2
python "heatmap coding2.py" path/to/results.xlsx --method spearman --output outputs/correlation.png
```

Kappa 的两组列分别按原行成对剔除缺失；未传序数标签时不计算加权 κ。热图仅适用于有意义的数值/有序编码，相关不等于一致性；默认 Spearman，可显式选择历史 Pearson。历史图片保持原样。

## 下一步实验

已实现工具可用于新收集数据；以下属于**待开展研究**：

1. 在同一任务、同一时间窗和模型版本下，随机化四条件的提问顺序；固定提示词与评分规则。
2. 记录搜索引用、原始输出、拒答和解析失败，并保留所有样本，避免选择性删除错误回答。
3. 对多模型、多次采样、主题或时间外测试扩展时，采用按样本/事件聚类的不确定性估计；当前 CLI 不支持这种嵌套设计。
4. 将准确率收益与成本、延迟及引用质量共同评估；这些指标尚未在当前代码实现。

详细公式、适用边界与复现材料清单见[实验协议](docs/protocol.md)。请引用上面的论文 DOI，并在使用新增工具时注明仓库 commit；不要把扩展指标描述成论文已发表结论。
