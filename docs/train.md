# Training MUPA

## 🛠️ Environment Setup

Ensure your system meets the following requirements:

* **CUDA** 11.8 / **CANN** 8.0
* **Python** 3.11.0
* **PyTorch** 2.4.0 / **Torch-NPU** 2.4.0.post2
* **Transformers** 4.45.2
* **DeepSpeed** 0.15.4
* **NNCore** 0.4.5

```bash
# Clone and install
git clone https://github.com/soughtlin/MUPA.git
cd MUPA
conda create -n mupa python=3.11 -y
conda activate mupa
pip install -r requirements.txt
```

For Ascend NPU users, adjust the NPU-specific dependencies in `requirements.txt` as needed.

## 🔑 Prepare Base Models

Download the following pretrained checkpoints from Hugging Face and place them under `model_zoo/`:

```bash
MUPA
└─ model_zoo
   ├─ Qwen2-VL-2B-Instruct
   └─ Qwen2-VL-7B-Instruct
```

* **Qwen2-VL-2B-Instruct**: [https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
* **Qwen2-VL-7B-Instruct**: [https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

## 📦 Dataset Preparation

We train three roles: **Grounder**, **Verifier**, and **GQA** (Grounded QA). Download and preprocess the datasets from our MUPA-Dataset release.

| Role     | Datasets                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------- |
| Grounder | `qvhighlights`, `didemo`, `tacos`, `queryd`, `cosmo_cap`, `internvid_vtime`, `hirest_grounding`, `hirest_step` |
| Verifier | `verifying`                                                                                                    |
| GQA      | `QVHighlights-QA`, `TACoS-QA`, `CosMo-Cap-QA`, `DeVE-QA`                                                       |

After extraction, your `data/` directory should look like:

```bash
MUPA
└─ data
   ├─ qvhighlights
   ├─ didemo
   ├─ tacos
   ├─ queryd
   ├─ cosmo_cap
   ├─ internvid_vtime
   ├─ hirest_grounding
   ├─ hirest_step
   ├─ verifying
   ├─ QVHighlights-QA
   ├─ TACoS-QA
   ├─ CosMo-Cap-QA
   └─ DeVE-QA
```

### Dataset for GQA

The **QVHighlights-QA**, **TACoS-QA**, and **CosMo-Cap-QA** subsets are derived from QVHighlights, TACoS, and CosMo-Cap}, respectively. We use **GPT-4o-mini** as a prior-knowledge assistant to generate question–answer pairs from existing video descriptions. A **BlindQA** filtering step then constrains each sample to 4–6 answer choices and evaluates the model without video input, yielding a baseline accuracy of ~20%. “BlindQA” serves to assess dataset bias and measure the gain from visual grounding.

## 🕹️ Start Training

We recommend training on **4 × NVIDIA H800 (80 GB)** or **Ascend 910B (65 GB)** devices. Adjust `nproc_per_node`, `per_device_train_batch_size`, and `gradient_accumulation_steps` to maintain a global batch size of 32 if your hardware differs.

```bash
# 1. Pretrain Grounder (2B / 7B)
bash scripts/pretrain/pretrain_grounder_2b.sh
bash scripts/pretrain/pretrain_grounder_7b.sh

# 2. Pretrain Verifier (2B / 7B)
bash scripts/pretrain/pretrain_verifier_2b.sh
bash scripts/pretrain/pretrain_verifier_7b.sh

# 3. Pretrain on GQA datasets (2B / 7B)
bash scripts/train/pretrain_gqa_2b.sh
bash scripts/train/pretrain_gqa_7b.sh
```

All logs and checkpoints will be written to `work_dirs/`. After training, you can update evaluation scripts with the new checkpoints for downstream testing.
