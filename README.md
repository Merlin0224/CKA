
# MLLM-Saliency-Steering: 深度多模态大模型幻觉抑制与表征对齐研究

本仓库提供了针对多模态大模型（MLLM，如 Qwen2-VL）视觉表征在深层网络中**坍塌（Representation Collapse）**现象的分析与干预框架。通过引入 **Channel-wise Saliency Gate (通道显著性门控)** 与 **Activation Steering (激活干预)** 技术，我们实现了在无需大规模微调的情况下，有效抑制多模态幻觉，并提升了模型的语义识别能力。

## 🔬 研究背景与动机
当前 MLLM（如 Qwen2-VL）虽然拥有强大的基座能力，但仍存在显著的视觉幻觉问题。本研究通过探针技术（Hooking）揭示了：视觉特征在进入大模型深层后，与初始输入的几何相似度（CKA）呈现显著衰减，这表明模型在推理时往往依赖“语言先验”而非“视觉事实”。

## 🛠 技术核心 (Key Innovations)
1. **通道显著性门控 (Channel-wise Saliency Gate)**：通过引入自适应门控机制，从视觉投影层（Merger）源头抑制背景噪声，放大关键语义通道。
2. **表征激活干预 (Activation Steering)**：无需参数更新，直接在 LLM 推理阶段（如 Layer 6-14）通过几何投影，将模型潜空间的“幻觉冲动”向“视觉忠实方向”平移。
3. **层级动力学分析 (Layer-wise Dynamical Analysis)**：系统化扫描不同层级对幻觉的影响，量化了 MLLM 内部的“认知对齐视界 (Critical Horizon)”。

## 📊 实验数据摘要 (基于 Flickr30k-POPE 基准)
通过对模型进行全深度扫描与干预，我们获得了如下核心实验结论：

| 实验组 | 幻觉率 (FPR) ↓ | 识别准确率 (Acc) ↑ | F1-Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Vanilla)** | 20.00% | 81.05% | 0.82 |
| **Proposed (Layer 6  Alpha 10.0)** | **15.56%** | **84.21%** | **0.85** |

*数据说明：实验基于 Flickr30k 测试集，采用 POPE 协议，验证了所提方法在抑制幻觉的同时显著增强了语义识别准确性。*

## 🚀 快速开始

### 1. 环境依赖
```bash
# 核心计算依赖
pip install torch transformers modelscope spacy pandas pyarrow
# 下载 NLP 基础模型用于实体识别
python -m spacy download en_core_web_sm
```

### 2. 实验运行
运行全层级扫描与对比实验脚本：
```bash
python -m test.eval_model
```
实验结果将自动保存至当前目录下的 `steering_report_YYYYMMDD-HHMMSS.csv`，其中记录了不同层级深度与干预强度 (Alpha) 对幻觉指标的影响。

## 📝 科研总结与反思
本研究通过量化的几何手段（CKA 与 Steering Vector）揭示了 MLLM 的视觉感知并非随着层数线性增强，而是存在非线性的“坍塌-重建”动力学。通过实验证明，在浅层（Layer 6）进行显著性注入是抑制幻觉的最佳帕累托边界。
