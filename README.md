# TinyTale-15M: Custom LLM Pre-training Pipeline

TinyTale-15M is an end-to-end engineering pipeline designed to pre-train a 15 Million parameter custom GPT-2 language model completely from scratch. The codebase handles autonomous offline tokenization, dynamic dataset subsetting, and managed training loops with automated safety checkpointing.

The companion trained weights are hosted natively on the Hugging Face Hub at [agentbyumer/TinyTale-15M](https://huggingface.co/agentbyumer/TinyTale-15M).

---

## 🛠️ System Architecture & Stack

- **Core Framework:** PyTorch & Hugging Face (Transformers, Accelerate, Datasets)
- **Package Management:** `uv` (Fast Python packaging engine)
- **Infrastructure:** 1x NVIDIA A10 GPU Workstation
- **Base Dataset:** `roneneldan/TinyStories` (200,000 unique story subset)

---

## 📦 Project Directory Layout

```text
TinyTale-15M/
├── .venv/                  # Isolated Python 3.12 environment
├── .gitignore              # Keeps model shards & binary caches out of Git
├── LICENSE                 # Open-source MIT License
├── README.md               # GitHub engineering documentation
├── main.py                 # Core orchestration script (Tokenizer + Trainer)
├── tokenized_dataset/      # Cached tokenized dataset (generated offline)
└── tiny_tale_checkpoints/  # Automated safety checkpoint shards
```

## 🚀 Step-by-Step Replication Guide
### 1. Environment Setup
Clone this repository to your local GPU instance, then leverage ```uv``` to build the isolated Python 3.12 virtual environment and spin up the CUDA-enabled dependencies:

```python
# Create the environment with Python 3.12
uv venv --python 3.12
.venv\Scripts\activate

# Install explicit CUDA 12.1 PyTorch binaries bypassing local caches
uv pip install torch torchvision torchaudio --index https://download.pytorch.org/whl/cu121 --no-cache

# Install remaining pipeline dependencies
uv pip install transformers datasets accelerate huggingface_hub
```

### 2. Execution
Run the complete training pipeline. The script automatically checks for an accelerated CUDA hardware mapping before launching:

```bash
python main.py
```

## 📊 Pipeline Processing Stages
1. **Phase 1 (Data Fetching):** Downloads the raw TinyStories dataset and selects a distinct 200,000 example slice to ensure linguistic diversity.

2. **Phase 2 (Offline Tokenization):** Runs a high-velocity dataset map processing tokens at over 600,000 examples/sec, caching the output to disk under ./tokenized_dataset.

3. **Phase 3 (Architecture Initialization):** Initializes a custom 6-layer, 6-attention-head GPT-2 matrix structure from scratch.

4. **Phase 4 (Training Execution):** Runs a 3,000-step training loop utilizing FP16 mixed precision, completing an epoch in roughly 14 minutes at ~3.44 iterations/sec.

## 📝 Generated Sample Inference Output

```Plaintext
[Prompt]: Once upon a time, a small puppy found a shiny key
[Output]: Once upon a time, a small puppy found a shiny key. The puppy was so excited! He couldn't wait to open it. He opened it to the corner of the key and opened it. He was so happy, he didn't know what to open it.
```

## 📜 License
This software pipeline is open-source and licensed under the [MIT License](./LICENSE).