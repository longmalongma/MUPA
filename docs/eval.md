# Evaluation Guide for MUPA

This document describes how to evaluate the MUPA models using our multipath evaluation scripts.

## 🛠 Environment Setup

Ensure you have installed all dependencies as described in [train.md](train.md).

## 📚 Checkpoint Preparation

Download the [base models](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d) and [MUPA checkpoints](https://huggingface.co/collections/soughtlin/mupa-683040ce74299f3194c89005), and place them into the `model_zoo` folder.

Your directory should look like:

```bash
MUPA
└─ model_zoo
   ├─ Qwen2-VL-2B-Instruct
   ├─ Qwen2-VL-7B-Instruct
   ├─ MUPA-2B
   └─ MUPA-7B
```


## 📦 Dataset Preparation

Arrange your `data/` folder as follows for each benchmark (example for NExT-GQA):

```bash
MUPA
└─ data
   └─ nextgqa
      ├─ videos/
      ├─ annotations.json
      └─ splits.txt
```

Supported benchmarks:

* `nextgqa` (NExT-GQA)
* `deve_qa` (DeVE-QA)
* `activitynet_captions` (ActivityNet-Captions)
* `tacos` (TACoS)
* `activitynet_rtl` (ActivityNet-RTL)

## 🔮 Running Evaluation

We provide two evaluation scripts that run the multi-path evaluation for 2B and 7B models respectively.

```bash
# Evaluate with the 2B model
bash run_scripts/eval_multipath_2b.sh <dataset> <task>

# Evaluate with the 7B model
bash run_scripts/eval_multipath_7b.sh <dataset> <task>
```

* `<dataset>` should be one of:`nextgqa`, `deveqa`, `activitynet_captions`, `tacos`, `activitynet_rtl`
* `<task>` must be `GQA` or `MR`

Example:

```bash
bash run_scripts/eval_multipath_2b.sh nextgqa GQA
bash run_scripts/eval_multipath_7b.sh tacos MR
```

Outputs and metrics will be saved in `outputs/` directories.

