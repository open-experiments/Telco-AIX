# Getting Started with AI-RAN Energy Optimization

This guide will help you get started with the AI-RAN Energy Efficiency Optimization project.

## Prerequisites

- Python 3.8+
- pip
- 4GB+ RAM recommended
- GPU optional (JAX will auto-detect and use if available)

## Installation

### 1. Clone and Navigate

```bash
cd Telco-AIX/airan-energy
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start (5 minutes)

Run the complete demo workflow:

```bash
python quickstart.py
```

This will:
1. Generate a synthetic dataset (10 cells × 30 days)
2. Train a traffic forecasting model (5 epochs)
3. Calculate energy savings
4. Display comprehensive results

Expected output:
- Energy savings: **20-35%**
- Cost reduction: **$15-30/day** per cell
- CO2 reduction: **5-15 kg/day** per cell

## Step-by-Step Guide

### Step 1: Generate Dataset

Create synthetic cell traffic data:

```bash
python src/data/dataset_generator.py \
    --num-cells 100 \
    --num-days 30 \
    --output-dir data
```

Output: `data/cell_traffic_100cells_30days.csv`

### Step 2: Train Traffic Forecaster

Train the JAX-based forecasting model:

```bash
python src/training/train_forecaster.py \
    --data-path data/cell_traffic_100cells_30days.csv \
    --output-path models/forecaster.pkl \
    --epochs 50 \
    --batch-size 32 \
    --plot
```

Training time:
- CPU: ~15-20 minutes
- GPU: ~5-7 minutes

### Step 3: Launch Dashboard

Visualize results in real-time:

```bash
streamlit run src/dashboard/app.py --server.port 8050
```

Open browser to: `http://localhost:8050`

### Step 4: Evaluate Performance

Test the trained model:

```python
from src.models.traffic_forecaster import TrafficForecasterWrapper
import numpy as np

# Load model
forecaster = TrafficForecasterWrapper()
forecaster.load_checkpoint('models/forecaster.pkl')

# Make prediction
test_input = np.random.randn(1, 168, 4)  # 7 days × 4 features
forecast = forecaster.forecast(test_input)

print(f"Forecast shape: {forecast.shape}")  # (24, 1) - next 24 hours
```

## Advanced Usage

### Training DQN Controller (Coming Soon)

```bash
python src/training/train_dqn.py \
    --forecaster-path models/forecaster.pkl \
    --episodes 1000 \
    --output-path models/dqn_controller.pkl
```

### Hyperparameter Tuning

Edit training scripts or use command-line args:

```bash
python src/training/train_forecaster.py \
    --learning-rate 5e-4 \
    --batch-size 64 \
    --epochs 100 \
    --patience 15
```

### Custom Dataset

To use your own cell traffic data, create a CSV with these columns:

| Column | Description | Type |
|--------|-------------|------|
| `timestamp` | Date/time | datetime |
| `cell_id` | Cell identifier | string |
| `traffic_mbps` | Traffic load in Mbps | float |
| `num_users` | Active users | int |
| `qos_score` | QoS score (0-100) | float |
| `capacity_mbps` | Cell capacity | float |
| `utilization` | Utilization % | float |

Then run:

```bash
python src/training/train_forecaster.py --data-path /path/to/your/data.csv
```

## Understanding the Output

### Energy Report

```
=== BASELINE (Always-On) ===
Total Energy: 240.50 kWh
Electricity Cost: $28.86
CO2 Emissions: 120.25 kg

=== OPTIMIZED (Sleep Strategy) ===
Total Energy: 168.35 kWh  ← 30% reduction
Electricity Cost: $20.20  ← 30% savings
CO2 Emissions: 84.18 kg   ← 30% reduction

=== SAVINGS ===
Energy Saved: 72.15 kWh (30.0%)
Cost Saved: $8.66 (30.0%)
CO2 Reduced: 36.08 kg (30.0%)
```

### Performance Metrics

The model tracks:
- **MAE (Mean Absolute Error)**: Traffic prediction accuracy in Mbps
- **QoS Impact**: Change in quality of service
- **Transition Count**: Number of sleep/wake cycles
- **Constraint Violations**: QoS threshold breaches

## Troubleshooting

### Issue: JAX not using GPU

```bash
# Check JAX GPU availability
python -c "import jax; print(jax.devices())"

# Install GPU version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: Out of Memory

Reduce batch size:

```bash
python src/training/train_forecaster.py --batch-size 16
```

### Issue: Training too slow

Use smaller lookback window:

```bash
python src/training/train_forecaster.py --lookback-window 48  # 2 days instead of 7
```

## Next Steps

1. **Experiment with Different Architectures**: Modify `src/models/traffic_forecaster.py`
2. **Implement DQN Training**: Complete the DQN training pipeline
3. **Real-world Data**: Integrate with O-RAN or vendor APIs
4. **Deployment**: Containerize for production (Kubernetes/OpenShift)
5. **Benchmarking**: Compare against PyTorch/TensorFlow baselines

## Project Structure

```
airan-energy/
├── src/
│   ├── data/
│   │   └── dataset_generator.py     ← Generate synthetic data
│   ├── models/
│   │   ├── traffic_forecaster.py    ← JAX forecasting model
│   │   ├── dqn_controller.py        ← DQN sleep controller
│   │   └── energy_calculator.py     ← Energy metrics
│   ├── training/
│   │   ├── train_forecaster.py      ← Training scripts
│   │   └── train_dqn.py
│   └── dashboard/
│       └── app.py                   ← Streamlit dashboard
├── notebooks/                       ← Jupyter notebooks
├── data/                            ← Generated datasets
├── models/                          ← Saved checkpoints
├── quickstart.py                    ← Quick demo script
├── requirements.txt
└── README.md
```

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Neural Networks](https://flax.readthedocs.io/)
- [O-RAN Alliance](https://www.o-ran.org/)
- [Telco-AIX Project](https://github.com/tme-osx/Telco-AIX)

## Support

For questions or issues:
1. Check the [main README](README.md)
2. Review [troubleshooting](#troubleshooting) section
3. Open an issue on GitHub

## License

See main repository LICENSE file.
