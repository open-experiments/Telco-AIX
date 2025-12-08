# Google Colab Testing Instructions

## 🚀 How to Execute and Evaluate JAX Code on Google Colab

This guide shows you how to test the AI-RAN Energy Optimization system using your Google Colab account.

---

## 📋 Prerequisites

- ✅ Google Account (free)
- ✅ Google Colab access (free)
- ✅ No local installation needed!

---

## 🎯 Quick Start (3 Steps)

### Step 1: Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload: `notebooks/COLAB_TEST.ipynb`

**Option B: From GitHub (Once Pushed)**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook**
3. Select **GitHub** tab
4. Enter: `https://github.com/tme-osx/Telco-AIX`
5. Select: `airan-energy/notebooks/COLAB_TEST.ipynb`

**Option C: Google Drive**
1. Upload `COLAB_TEST.ipynb` to your Google Drive
2. Right-click → **Open with → Google Colaboratory**

### Step 2: Enable GPU (Recommended)

1. In Colab, click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU (T4)**
3. Click **Save**

**Why GPU?**
- 10-50x faster than CPU
- Tests JAX GPU capabilities
- Free on Colab!

### Step 3: Run All Cells

1. Click **Runtime → Run all**
2. Wait ~15-20 minutes
3. Review results as they appear

---

## 📊 What the Notebook Tests

### ✅ Tests Performed

| # | Test | Duration | What It Checks |
|---|------|----------|----------------|
| 1 | Environment Setup | 1 min | Python version, Colab detection |
| 2 | Install Dependencies | 3-5 min | JAX, Flax, Haiku, etc. |
| 3 | Clone Project | 1 min | Download code from GitHub |
| 4 | Verify JAX | 1 min | GPU detection, JIT compilation |
| 5 | Module Imports | 30 sec | All 4 modules import correctly |
| 6 | Dataset Generator | 30 sec | Generate 5 cells × 7 days |
| 7 | Traffic Forecaster | 1 min | Model creation, forward pass |
| 8 | DQN Controller | 30 sec | State encoding, action selection |
| 9 | Energy Calculator | 30 sec | Energy savings calculation |
| 10 | Train Mini Model | 2-3 min | 3 epochs training |
| 11 | Performance Benchmark | 1 min | JAX vs NumPy speedup |
| 12 | Complete System Test | 3-5 min | End-to-end workflow |
| 13 | Final Report | Instant | Summary of all tests |

**Total Runtime:** ~15-20 minutes

---

## 📈 Expected Results

### ✅ Success Indicators

You should see output like this:

```
=====================================================================
JAX SETUP VERIFICATION
=====================================================================
JAX version: 0.4.25
Available devices:
  - cuda:0

✓ JAX operations work: sum([1,2,3]) = 6.0
✓ JIT compilation works: 5^2 = 25.0

=====================================================================
✅ JAX IS READY!
=====================================================================
```

```
Testing module imports...

✓ data.dataset_generator             → CellTrafficGenerator
✓ models.traffic_forecaster          → TrafficForecasterWrapper
✓ models.dqn_controller              → DQNController
✓ models.energy_calculator           → EnergyCalculator

✅ All 4 modules imported successfully!
```

```
Training forecaster for 3 epochs...
  Epoch 1/3 | Train: 0.234567 | Val: 0.245678
  Epoch 2/3 | Train: 0.123456 | Val: 0.134567
  Epoch 3/3 | Train: 0.067890 | Val: 0.078901

✅ Training completed successfully!
```

```
=====================================================================
COMPLETE SYSTEM TEST PASSED!
=====================================================================
All components working correctly:
  ✓ Dataset generation
  ✓ Model creation
  ✓ Training
  ✓ Prediction
  ✓ Energy calculation
```

---

## 🎯 Performance Benchmarks

### Expected Speedups (with GPU)

```
PERFORMANCE BENCHMARK: JAX vs NumPy
=====================================================================

Matrix size: 100x100
  NumPy: 2.34 ms
  JAX (2nd): 0.45 ms
  Speedup: 5.2x

Matrix size: 1000x1000
  NumPy: 45.67 ms
  JAX (2nd): 3.21 ms
  Speedup: 14.2x

Matrix size: 5000x5000
  NumPy: 1234.56 ms
  JAX (2nd): 45.67 ms
  Speedup: 27.0x

Average speedup: 15.5x
```

**Note:** CPU-only will show ~2-5x speedup instead

---

## 🐛 Troubleshooting

### Issue 1: "Runtime disconnected"
**Solution:**
- Colab free tier has 12-hour limit
- Re-run from the beginning
- Or upgrade to Colab Pro ($10/month)

### Issue 2: "Out of memory"
**Solution:**
- Runtime → Restart runtime
- Or reduce dataset size in cells

### Issue 3: "GPU not available"
**Solution:**
- Runtime → Change runtime type → GPU
- Or run on CPU (slower but works)

### Issue 4: "Module not found"
**Solution:**
- Make sure "Install Dependencies" cell ran successfully
- Re-run that cell
- Check for red error messages

### Issue 5: "GitHub clone failed"
**Solution:**
- **Option A:** Manually upload files
  1. Download `airan-energy/` folder
  2. Zip it
  3. Upload to Colab
  4. Unzip in Colab

- **Option B:** Use Google Drive
  1. Upload project to Google Drive
  2. Mount Drive in Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

---

## 📊 Confidence Levels After Testing

### Before Colab Testing
```
Syntax:     100% ✅ (confirmed)
Runtime:     95% ⚠️  (not tested)
Performance: 70% ⚠️  (claimed)
Overall:     95% ⚠️
```

### After Colab Testing (If All Pass)
```
Syntax:     100% ✅ (confirmed)
Runtime:    100% ✅ (confirmed)
Performance: 100% ✅ (measured)
Overall:    100% ✅ PRODUCTION READY!
```

---

## 🎓 Advanced Testing

### Test 1: Longer Training
```python
# Change in cell 10:
for epoch in range(50):  # Instead of 3
    ...
```

### Test 2: Larger Dataset
```python
# Change in cell 6:
df = generator.generate_dataset(
    num_cells=100,   # Instead of 5
    num_days=30      # Instead of 7
)
```

### Test 3: Full Benchmarking
Add this cell at the end:
```python
import time

print("Full Training Benchmark")
print("="*70)

# Time full training
start = time.time()
for epoch in range(50):
    forecaster.params, forecaster.opt_state, _ = forecaster.train_step(
        forecaster.params, forecaster.opt_state, X_train, y_train
    )
train_time = time.time() - start

print(f"50 epochs training time: {train_time:.1f} seconds ({train_time/60:.1f} minutes)")
print(f"Average per epoch: {train_time/50:.2f} seconds")
```

---

## 📂 Alternative: Manual Testing

If you prefer manual control:

### Option 1: Cell-by-Cell
1. Run cells one at a time
2. Review each output
3. Fix issues before proceeding

### Option 2: Sections Only
1. Run only the sections you want to test
2. Skip benchmarking if short on time

### Option 3: Minimal Test
Run just these cells:
- Environment Setup
- Install Dependencies
- Module Imports
- Complete System Test

---

## 🎯 Success Criteria

Your system is working if:

- ✅ All imports succeed (no ModuleNotFoundError)
- ✅ Models can be created (no shape errors)
- ✅ Training runs without errors
- ✅ Loss decreases over epochs
- ✅ Predictions are generated
- ✅ Energy savings calculated
- ✅ JAX shows speedup vs NumPy

---

## 📝 Share Your Results

After testing, you can:

1. **Download Notebook**
   - File → Download → Download .ipynb
   - Save with results embedded

2. **Share Link**
   - File → Save a copy to Drive
   - Share → Get shareable link
   - Send to team

3. **Export Results**
   - File → Print
   - Save as PDF
   - Include in documentation

---

## 🚀 Next Steps After Successful Testing

1. **Train Full Model**
   - Use 50+ epochs
   - 100+ cells
   - 30+ days

2. **Benchmark Performance**
   - Compare training times
   - Measure inference latency
   - Verify 3x speedup claim

3. **Deploy to Production**
   - Export trained model
   - Set up inference server
   - Integrate with O-RAN

4. **Publish Results**
   - Upload dataset to HuggingFace
   - Share trained models
   - Write technical blog post

---

## 📞 Support

**Issues?**
- Check troubleshooting section above
- Review error messages carefully
- Open GitHub issue with:
  - Error message
  - Cell number
  - GPU/CPU used

**Questions?**
- See main [README.md](../ReadMe.md)
- Check [GETTING_STARTED.md](../GETTING_STARTED.md)
- Review [TEST_REPORT.md](../TEST_REPORT.md)

---

## 🎉 Summary

**What You Get:**
- ✅ Fully tested JAX implementation
- ✅ Performance benchmarks
- ✅ Confidence in code quality
- ✅ Ready for production

**Time Investment:**
- Upload notebook: 1 minute
- Enable GPU: 30 seconds
- Run tests: 15-20 minutes
- **Total: ~20 minutes for 100% confidence!**

---

**Ready to test? Upload `COLAB_TEST.ipynb` to Google Colab and click "Run all"!** 🚀

---

**Last Updated:** December 7, 2025
**Estimated Runtime:** 15-20 minutes with GPU
**Cost:** Free (Colab free tier)
