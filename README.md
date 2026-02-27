# A-MoE-VLM: Logic-Aligned Compact Vision-Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/Framework-ms--swift-orange.svg)](https://github.com/modelscope/swift)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official implementation of the **Logic-V** alignment pipeline for highly compact Vision-Language Models. This repository provides the complete training logic, dynamic ORB keyframe extraction tools, and pre-trained LoRA weights to reproduce our state-of-the-art results on complex video reasoning tasks.

By utilizing a "Distillation-then-RL" paradigm with a 72B-parameter teacher model, our highly compact **3B-parameter** model achieves remarkable reasoning capabilities, significantly closing the gap with massive proprietary models.

## 🌟 Key Results (EgoSchema)
- **Single Expert (Logic-V RL)**: 59.0% Accuracy
*(Note: To emphasize the core contribution of the Logic-V reward mechanism, this repository highlights the single-expert 3B model. Our full A-MoE framework builds upon this to achieve 60.8% accuracy.)*

---

## 📂 Repository Structure

```text
A-MoE-VLM/
├── checkpoint/           # Pre-trained LoRA weights (Download required)
├── dataset/              # Subsets and JSON annotations for evaluation
├── evaluation/           # Raw prediction logs (3B-RL & 72B-Teacher) for transparency
├── inference/            # Scripts for reproducing the 59.0% accuracy
└── scripts/              # RL training launcher, Logic-V rewards, and 72B analysis tools
```

---

## 🚀 1. Quick Start & Installation

Clone this repository and set up the environment. We highly recommend using Python 3.10+ and CUDA 11.8+.

```bash
git clone https://github.com/bitxige/A-MoE-VLM.git
cd A-MoE-VLM
```

**Core Environment Dependencies:**
- `Python` >= 3.10
- `Transformers` >= 4.40.0
- `ms-swift` (for RLHF and LoRA training)
- `DeepSpeed` >= 0.14.0 (for RL acceleration)
- `qwen-vl-utils` & `opencv-python` (for visual processing)

You can easily install the core dependencies via:
```bash
pip install -r requirements.txt
```

---

## 📥 2. Download Pre-trained Weights

To facilitate immediate reproduction, we fully open-source our 5600-step RL LoRA weights. You do not need to retrain the model from scratch.

Download the `adapter_model.safetensors` and place it in the `checkpoint/checkpoint-5600/` directory:

```bash
mkdir -p checkpoint/checkpoint-5600
# Download the core weights (228MB)
wget https://github.com/bitxige/A-MoE-VLM/releases/download/v1.0/adapter_model.safetensors -O checkpoint/checkpoint-5600/adapter_model.safetensors
```

> ⚠️ **Note:** Ensure that the accompanying `adapter_config.json` is also placed in the same directory. If your local folder name differs, please adjust the paths accordingly.

---

## 📊 3. Reproducing Evaluations (Inference)

All evaluation scripts are neatly organized in the `inference/` directory.

> ⚠️ **Path & Filename Notice:** The exact filenames for your scripts (e.g., `test.py` vs `eval_model.py`) or datasets might vary based on your local setup. **Please verify and replace the placeholder paths in the commands below with your actual local file paths before running.**

### Evaluate Single RL Expert (~59.0%)
Runs the single generalist expert trained via the Logic-V reward pipeline.

```bash
python inference/test.py \
    --base_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --model_path "checkpoint/checkpoint-5600" \
    --input_file "dataset/egoschema_subset.jsonl" \
    --output_file "evaluation/RL-egoschema.jsonl"
```

### Evaluate SFT Baseline
For performance comparison against the standard supervised fine-tuned model:

```bash
python inference/eval_sft.py \
    --base_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --lora_path "checkpoint/Qwen2.5-3B-Video-SFT/checkpoint-504" \
    --input_file "dataset/egoschema_subset.jsonl" \
    --output_file "evaluation/sft_egoschema_predictions.jsonl" \
    --max_new_tokens 1024
```

---

## 🧠 4. Teacher Annotations & Logic-V Reward

We provide the exact scripts used to generate the logical reasoning anchors and dynamic visual features from the 72B teacher model.

### Extract Visual Logs & ORB Keyframes (72B-Int4)
This script utilizes an ORB-based homography extraction pipeline to filter redundant frames and uses Qwen2.5-72B to build chronological semantic logs.

```bash
# Example usage for extracting teacher logs
python scripts/analyze_72b.py \
    --model_path "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
    --video_dir "dataset/videos" \
    --input_json "dataset/egoschema.json" \
    --output_file "evaluation/egoschema_72b_embodied_r_analysis.jsonl" \
    --max_frames 24
```

### Logic-V Reward Engineering
The core logic for our RL pipeline (checking logical consistency via external arbiters and formatting) is located in the `scripts/train/` directory (e.g., `rewards.py`).

---

## 🛡️ Reproducibility Note

The reported accuracy (59.0% for the RL single-expert) was evaluated using `bfloat16` precision on a **single 48GB GPU** (e.g., vGPU-48GB / RTX 4090 / A800), which is highly efficient and more than sufficient for inference. *(Note: During the Reinforcement Learning phase, a dual-GPU setup was utilized solely to accelerate the training process.)*

Please note that due to the inherent non-determinism in parallel CUDA operations, architectural variations in Bfloat16 matrix multiplications across different GPU models, and specific environment configurations (e.g., CUDA 11.8, PyTorch, Transformers versions), **you may observe natural fluctuations in the final reproduced accuracy.** To ensure absolute transparency and academic integrity, we have provided our **exact raw prediction logs** containing the complete `<think>` traces in the `evaluation/` directory, serving as the definitive ground-truth for our reported metrics.

---

## 🤝 Acknowledgments
This code is built upon the excellent [ms-swift](https://github.com/modelscope/swift) framework for LLM/VLM fine-tuning and RLHF.
