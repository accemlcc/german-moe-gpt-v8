# German MoE GPT v8 - OPUS EDITION

A research-grade language model with state-of-the-art Mixture-of-Experts (MoE) architecture, trained on consumer hardware (RTX 4090). This implementation follows best practices from recent MoE research (ST-MoE, Switch Transformer) while maintaining full cross-platform compatibility.

> **Note:** While this model was trained on German data, the architecture is language-agnostic and can be used for any language dataset. Simply replace the training corpus with your target language data.

## Project Status (October 2025)

-   **v8 Pre-Training:** ✅ **COMPLETE**
-   **Fine-Tuning Phase:** 🔬 **IN PROGRESS**

## Overview

Development of a high-performance language model with state-of-the-art MoE architecture on a single consumer GPU. The v8 model was trained on a 17.4 GB high-quality German corpus and demonstrates strong coherence without SEO spam artifacts.

## 🏗️ Architecture

### Model Specifications

-   **Total Parameters:** 149.6M
-   **Active Parameters per Token:** ~49.9M (~33%)
-   **Architecture:** Hybrid Dense + MoE Transformer
-   **Experts per MoE Layer:** 32
-   **Active Experts (Top-k):** 2
-   **Context Length:** 2048 Tokens
-   **Vocabulary:** 128,256 (Llama 3.2 Tokenizer)

### Core Components

#### 1. **Mixture-of-Experts Layer**
- **Noisy Top-k Router** with learnable gating mechanism
- **Dynamic Expert Capacity Management** to prevent token overflow
- **Load Balance Loss** (Switch Transformer) for uniform expert utilization
- **Router Z-Loss** (ST-MoE) for numerical stability
- **FP32 Router Computation** to avoid precision issues

#### 2. **Attention Mechanism**
- **Rotary Position Embeddings (RoPE)** instead of classical positional encodings
- **PyTorch SDPA** (Scaled Dot Product Attention) with automatic backend selection
- **Causal Masking** for autoregressive generation
- **Multi-Head Self-Attention** with 12 heads

#### 3. **Expert Architecture**
- **Batch Matrix Multiplication** for parallel expert processing
- **SwiGLU Activation** (optional, alongside GELU/ReLU)
- **4x Hidden Dimension** (standard for GPT architecture)
- **Shared Expert Weights** as 3D tensors for efficiency

#### 4. **HuggingFace Integration**
- Fully compatible with `transformers` library
- Inherits from `PreTrainedModel` and `GenerationMixin`
- Supports `.generate()` for inference
- **Weight Tying** between token embeddings and LM head
- **Gradient Checkpointing** support for memory efficiency

### Technical Features

#### 🔬 **Research-Backed Design**
- Implementation based on **ST-MoE** (Zoph et al. 2022) and **Switch Transformer** (Fedus et al. 2022)
- Auxiliary loss functions for stable MoE training
- Capacity factor management (1.25 training, 2.0 evaluation)
- Expert-specific initialization with fan-in scaling

#### ⚡ **Performance & Efficiency**
- **Mixed Dense + MoE Layers** (every 2nd layer is MoE) for optimal parameter utilization
- Batch-based expert processing (no iterative loops)
- Automatic SDPA backend optimization (Flash Attention when available)
- Gradient accumulation & mixed precision training support

#### 🖥️ **Cross-Platform Compatibility**
- Pure PyTorch implementation without external kernels
- Runs on **Windows, Linux, macOS**
- No CUDA-only dependencies (Liger, Flash Attention libraries)
- `pip install transformers torch` is sufficient for setup

#### 📊 **Monitoring & Debugging**
- TensorBoard integration for training metrics
- Aux loss & router z-loss tracking
- Sample generation callbacks during training
- Expert load distribution monitoring

## 📊 Training Details

### Dataset (v8 OPUS Mix - German)

-   **Clean German Wikipedia:** ~11 GB (encyclopedic knowledge)
-   **OpenSubtitles (German):** Dialog corpus (natural language)
-   **Belletristik:** German literature corpus (style & creativity)
-   **Total Size:** ~17.4 GB
-   **Quality:** Deduplicated, SEO spam filtered

> **Adapting to other languages:** Replace the dataset with your target language corpus. The architecture supports any tokenizer and language.

### Pre-Training Results

-   **Training Progress:** 300,000 / 300,000 steps
-   **Training Loss:** 12.0 → 2.55 (79% reduction)
-   **Validation Loss:** 4.58 → 2.40 (48% reduction)
-   **Final Perplexity:** **11.0** (exp(2.40))
-   **Total Training Time:** ~120 hours (RTX 4090)
-   **Hardware:** Single consumer GPU (24GB VRAM)

### Configuration

```python
# Architecture
n_layer = 12                    # Transformer blocks
n_embd = 768                    # Hidden dimension
n_head = 12                     # Attention heads
n_experts = 32                  # Experts per MoE layer
n_experts_active = 2            # Top-k routing
moe_layer_frequency = 2         # Every 2nd layer is MoE

# Training
batch_size = 32
gradient_accumulation_steps = 4
max_lr = 3e-4
capacity_factor = 1.25          # Expert capacity
aux_loss_alpha = 0.01           # Load balance loss weight
router_z_loss_alpha = 0.001     # Router z-loss weight
```

## 🚀 Usage

### Installation

```bash
# Create conda environment
conda create -n german_moe python=3.10
conda activate german_moe

# Install dependencies
pip install -r requirements.txt
```

### Start / Resume Training

The training script automatically detects existing checkpoints and resumes training:

```bash
python train_moe_v8_clean.py
```

**Key Features:**
- Automatic checkpoint recovery
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Sample generation during training
- TensorBoard logging

### Inference / Text Generation

```bash
python inference.py
```

**Example Usage:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("./moe_final_v8_clean")
tokenizer = AutoTokenizer.from_pretrained("./moe_final_v8_clean")

# Generate text
prompt = "Die Hauptstadt von Deutschland ist"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

### Monitoring

```bash
# Start TensorBoard
tensorboard --logdir=./logs_v8_clean

# or on Windows:
start_tensorboard.bat

# Watch generated samples
tail -f samples_v8_clean/generation_log.txt  # Linux/Mac
Get-Content samples_v8_clean/generation_log.txt -Wait  # Windows PowerShell

# Check GPU utilization
nvidia-smi -l 1
```

## 📁 Project Structure

```
german-moe-gpt-v8/
├── moe_model.py              # Main model definition
├── moe_layers.py             # MoE layer & router
├── moe_config.py             # Configuration (HF-compatible)
├── moe_trainer.py            # Custom trainer
├── train_moe_v8_clean.py     # Training script
├── inference.py              # Inference script
├── sample_generation_callback.py  # Training callback
├── moe_checkpoints_v8_clean/ # Training checkpoints
├── moe_final_v8_clean/       # Final models
├── logs_v8_clean/            # TensorBoard logs
└── samples_v8_clean/         # Generated text samples
```

## 🔬 Technical Details

### MoE Router Algorithm

The router uses a **Noisy Top-k Gating Mechanism**:

1. **Gate Computation:** `router_logits = W_gate @ hidden_states`
2. **Noise Injection (Training):** `router_logits += softplus(W_noise @ hidden_states) * ε`
3. **Top-k Selection:** Selects the k best experts per token
4. **Capacity Management:** Limits tokens per expert (prevents overload)
5. **Weighted Routing:** Tokens are routed to experts with weights

### Loss Functions

**Total Loss:**
```
L_total = L_ce + α * L_aux + β * L_z
```

- **L_ce:** Cross-entropy language modeling loss
- **L_aux:** Load balance loss (expert utilization)
- **L_z:** Router z-loss (numerical stability)
- **α = 0.01, β = 0.001:** Empirically optimized weights

### Memory Optimization

- **Gradient Checkpointing:** Reduces VRAM usage by ~40%
- **Mixed Precision (BF16):** 2x faster training
- **Gradient Accumulation:** Simulates larger batch sizes
- **Weight Tying:** LM head shares weights with token embeddings

## 📚 References

This project implements techniques from the following research papers:

- **ST-MoE:** [Zoph et al. 2022 - "Designing Effective Sparse Expert Models"](https://arxiv.org/abs/2202.08906)
- **Switch Transformer:** [Fedus et al. 2022 - "Switch Transformers"](https://arxiv.org/abs/2101.03961)
- **RoFormer:** [Su et al. 2021 - "RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864)

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- HuggingFace Transformers team for the excellent framework
- PyTorch team for SDPA and optimized operations
- nanoGPT/nanoMoE community for inspiration
