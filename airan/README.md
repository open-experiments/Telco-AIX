# AI-RAN: Neural Receiver for 5G PUSCH

**AI-enhanced 5G Physical Uplink Shared Channel (PUSCH) receiver using NVIDIA Sionna on Red Hat OpenShift AI**

Demonstrates how deep learning can replace conventional channel estimation + equalization + soft demapping in 5G uplink processing, achieving 2-3 dB SNR gains at BLER = 10⁻².

---

## What This Demonstrates?

- **Neural Receiver**: End-to-end learned receiver replacing conventional LS channel estimation + MMSE equalization + soft demapping
- **SNR Gain**: 2-3 dB improvement at BLER = 10⁻² compared to conventional receivers
- **Real-time Inference**: <1ms latency per PUSCH slot with TensorRT FP16 optimization
- **GPU-Optimized Pipeline**: Full training and inference pipeline on NVIDIA RTX 4090
- **OpenShift AI Integration**: Complete MLOps workflow on Red Hat OpenShift AI

---

## Tech-Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **5G Simulation** | NVIDIA Sionna | 1.2.1 | 3GPP-compliant channel modeling & data generation |
| **ML Framework** | TensorFlow | 2.20.0+ | Neural network training |
| **Inference Optimization** | TensorRT | 10.x | FP16 GPU inference (<1ms latency) |
| **Platform** | Red Hat OpenShift AI | 2.x | Jupyter notebooks, GPU scheduling, MLOps |
| **Hardware** | NVIDIA RTX 4090 D | 48GB VRAM | Ada Lovelace GPU architecture |
| **Optional** | NVIDIA Aerial SDK | 25.2 | cuBB integration (validation only) |

---

## Repo Structure

```
aerial-airan/
│
├── notebooks/
│   ├── 00-environment-validation.ipynb   # ✅ Verify Sionna 1.2.1 + TensorFlow + GPU
│   ├── 01-dataset-generation.ipynb       # ✅ GPU-optimized PUSCH data generation
│   ├── 02-train-neural-rx.ipynb          # ✅ Train ResNet+Attention neural receiver
│   ├── 03-optimize-tensorrt.ipynb        # ✅ Convert to TensorRT FP16
│   └── 04-validate-performance.ipynb     # ✅ Benchmark vs conventional receiver
│
├── setup.sh                               # One-time environment setup script
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## Quick Start

### Prerequisites

- **Red Hat OpenShift** cluster with GPU Operator installed
- **Red Hat OpenShift AI 2.x** (RHOAI)
- **NVIDIA GPU** (RTX 4090 recommended, or better A100/H100 if you have funds)
- **500GB storage** for datasets and models
- **oc CLI** configured with cluster access

### Step 1: Setup Environment

Open JupyterLab, upload the notebooks and setup files, then run:

```bash
# In a JupyterLab terminal
bash setup.sh
```

This installs:
- Sionna 1.2.1
- TensorFlow 2.20.0+
- h5py, matplotlib, seaborn, scipy, tqdm
- GPU memory growth configuration
- Workspace directories (`data/`, `models/`, `results/`)

### Step 2: Run the Notebooks

Execute notebooks **in order**:

| Notebook | Duration | Description |
|----------|----------|-------------|
| **00-environment-validation.ipynb** | ~2 min | Verify Sionna 1.2.1, TensorFlow, GPU detection |
| **01-dataset-generation.ipynb** | 30 min - 2 hrs | Generate training data (SMALL/MEDIUM/LARGE) |
| **02-train-neural-rx.ipynb** | ~2 hours | Train neural receiver (ResNet + Attention) |
| **03-optimize-tensorrt.ipynb** | ~10 min | Convert to TensorRT FP16 |
| **04-validate-performance.ipynb** | ~30 min | Measure BLER, compare with conventional RX |

---

## Dataset Generation Options

Notebook 01 provides **interactive dataset size selection**:

| Size | Samples | SNR Points | Duration | Disk Space | Use Case |
|------|---------|------------|----------|------------|----------|
| **SMALL** | 10,000 | 11 | ~30 min | ~43 GB | Quick testing, prototyping |
| **MEDIUM** | 50,000 | 15 | ~2.5 hrs | ~215 GB | Balanced training |
| **LARGE** | 100,000 | 21 | ~5 hrs | ~430 GB | Best accuracy, publication-quality |

**To select:** Edit `DATASET_CHOICE = 1` (SMALL) / `2` (MEDIUM) / `3` (LARGE) in notebook cell.

---

## Technical Details

### 5G PUSCH Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Carrier Frequency** | 3.5 GHz | 5G NR n78 band |
| **FFT Size** | 4096 | OFDM FFT length |
| **Subcarriers** | 896 effective | After DC & guard bands |
| **OFDM Symbols** | 14 | One slot duration |
| **Modulation** | 16-QAM | 4 bits per symbol |
| **MIMO** | SIMO 1×4 | 1 TX antenna, 4 RX antennas |
| **Channel Model** | Frequency-selective Rayleigh | 23-tap exponential power delay profile |

### Channel Model

**Custom Frequency-Selective Rayleigh Fading:**
- **23 taps** with exponential power delay profile: P(τ) ∝ e^(-τ/5)
- **Rayleigh fading** per tap: h_l ~ CN(0, P_l)
- **Frequency response** via DFT: H[f] = Σ h_l e^(-j2πfτ_l)
- **GPU-optimized** with `@tf.function` compilation and matrix multiplication

*Note: Originally planned to use 3GPP CDL-C Urban Macro, but switched to custom Rayleigh fading due to Sionna 1.2.1 API complexity.*

### Neural Receiver Architecture

**ResNet + Multi-Head Attention:**

```
Input: y (received signal)
  Shape: [batch, num_rx=4, num_subcarriers=896, num_symbols=14, 2]
        (real/imag components stacked)
    ↓
Spatial Processing (per-antenna Conv2D)
    ↓
ResNet Blocks (64 → 128 → 256 filters)
    ↓
Multi-Head Self-Attention (focus on reliable subcarriers)
    ↓
Global Average Pooling
    ↓
Dense Layers (512 → 512 → num_bits)
    ↓
Output: LLRs (Log-Likelihood Ratios)
  Shape: [batch, num_bits]
  Activation: tanh (bounded to [-1, +1])
```

**Key Features:**
- **Joint processing**: No separate channel estimation step
- **SNR-agnostic**: Trained across multiple SNR points (-10 to +10 dB)
- **Attention mechanism**: Learns to focus on reliable subcarriers
- **Skip connections**: Deep architecture (ResNet backbone) with gradient flow

**Training:**
- **Loss**: Binary cross-entropy with LLR outputs
- **Optimizer**: Adam (lr=1e-3 → 1e-6 with ReduceLROnPlateau)
- **Batch size**: 64
- **Epochs**: ~20 (with early stopping)
- **Data augmentation**: Multiple SNR points per batch

### TensorRT Optimization

**FP32 → FP16 Conversion:**
- **Precision**: Mixed FP16 inference
- **Layer fusion**: Conv+BN+ReLU → single CUDA kernel
- **Kernel auto-tuning**: RTX 4090-specific optimization
- **Speedup**: 5-10× faster than TensorFlow FP32
- **Target latency**: <1ms per PUSCH slot

---

## Expected Results

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Time** | ~2 hours | RTX 4090, SMALL dataset |
| **Inference Latency** | <1ms | TensorRT FP16, per slot |
| **SNR Gain @ BLER=10⁻²** | 2-3 dB | vs conventional LS+MMSE |
| **Throughput** | >1000 slots/sec | Batched inference |

### BLER Comparison (Expected)

```
SNR (dB)  | Conventional | Neural RX | Gain (dB)
----------|--------------|-----------|----------
   -4     |   ~0.12      |  ~0.03    |  ~3.5
   -2     |   ~0.05      |  ~0.01    |  ~2.8
    0     |   ~0.02      |  ~0.003   |  ~2.5
    2     |   ~0.005     |  ~0.0005  |  ~2.3
    4     |   ~0.001     |  <0.0001  |  >2.0
```

*Actual results depend on channel conditions and dataset size.*

---

## Recent Updates (November 2025)

### Sionna 1.2.1 Migration Complete

All notebooks updated for **Sionna 1.2.1** (from 0.18.x):

**Breaking API Changes:**
- Module structure: `sionna.*` → `sionna.phy.*`
- BinarySource removed → `sample_bernoulli([batch, num_bits], p=0.5)`
- CDL initialization: now requires `PanelArray` objects

**See:** [`SIONNA-1.2-API-CHANGES.md`](SIONNA-1.2-API-CHANGES.md) for full migration guide.

### GPU Optimization

- **@tf.function compilation** for channel generation
- **Matrix multiplication** instead of broadcast ops
- **Pre-computed channel parameters**
- **Result**: ~40s/batch (down from 60s/batch)

### Interactive Dataset Selection

- **User-friendly**: Simple variable assignment (no input() prompts)
- **Dynamic sizing**: Automatic batch calculation based on selection
- **Size estimates**: GB calculations for each option

---

## References

### Papers & Research

1. **S. Cammerer et al.**, "A Neural Receiver for 5G NR Multi-user MIMO"
   *IEEE Globecom 2023*
   [Paper](https://arxiv.org/abs/2312.02601) | [Code](https://github.com/NVlabs/neural_rx)

2. **T. Gruber et al.**, "On Deep Learning-Based Channel Decoding"
   *IEEE CISS 2017*
   [Paper](https://arxiv.org/abs/1701.07738)

### Documentation

1. **NVIDIA Sionna**
   [Website](https://nvlabs.github.io/sionna/) | [GitHub](https://github.com/nvlabs/sionna) | [API Docs](https://nvlabs.github.io/sionna/api/)

2. **NVIDIA Aerial SDK**
   [Docs](https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-2/) | [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/aerial)

3. **Red Hat OpenShift AI**
   [Product Page](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

---

## 🎉 Acknowledgments

- **NVIDIA Research** Sionna and neural_rx reference implementation
- **Apple Radio & Antenna Team** Data Generator and ETL.
- **Telco-AIX Team** Sandbox Env & Continous Support.

---

