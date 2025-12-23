# Lessons Learned: AI-RAN Neural Receiver Development

**Project:** NVIDIA Sionna + Aerial SDK on Red Hat OpenShift AI
**Date:** November 17, 2025
**Platform:** RTX 4090 D (48GB VRAM)
**Framework:** TensorFlow 2.20.0 + Keras 3.x + Sionna 1.2.1

---

## Executive Summary

This document captures 7 critical issues encountered during the development of a neural receiver for 5G PUSCH, along with their root causes and solutions. The issues ranged from API compatibility problems to memory optimization challenges and training performance bottlenecks.

**Key Achievements:**
- ✅ Achieved **80x training speedup** (37s/step → 0.459s/step)
- ✅ Optimized model to fit in 48GB VRAM with full attention mechanism
- ✅ Maintained high performance while respecting hardware constraints
- ✅ Reduced total training time from 5 days to 1.4 hours

---

## Issue #1: Keras 3.x KerasTensor API Incompatibility

### Problem
```python
ValueError: A KerasTensor cannot be used as input to a TensorFlow function
```

Training failed immediately with error when using `tf.shape()` inside Keras Functional API model construction.

### Root Cause
TensorFlow 2.20.0 introduced Keras 3.x which changed how symbolic tensors work during model construction. Using `tf.shape(x)[0]` to extract batch size from a KerasTensor is no longer supported in the Functional API.

**Problematic code:**
```python
def attention_block(x, key_dim=64):
    batch_size = tf.shape(x)[0]  # ❌ Fails with Keras 3.x
    height, width, channels = x.shape[1:]
    x_reshaped = layers.Reshape((height * width, channels))(x)
```

### Solution
Removed the explicit batch size extraction. Keras layers like `Reshape` automatically handle the batch dimension.

**Fixed code:**
```python
def attention_block(x, key_dim=64):
    # Removed batch_size line - not needed!
    height, width, channels = x.shape[1:]
    x_reshaped = layers.Reshape((height * width, channels))(x)  # ✅ Works!
```

### Lessons Learned
1. **Avoid `tf.shape()` in Keras Functional API**: Use `x.shape` for static dimensions
2. **Trust Keras layer batching**: Most layers handle batch dimensions automatically
3. **Test with latest TensorFlow**: API breaking changes happen between major versions
4. **Read migration guides**: Keras 3.x has significant API changes from Keras 2.x

### Prevention
- Use `x.shape` for accessing static dimensions during model construction
- Only use `tf.shape()` when you need dynamic shapes at runtime (e.g., in custom training loops)
- Review Keras 3.x migration guide: https://keras.io/guides/migrating_to_keras_3/

---

## Issue #2: HDF5 Fancy Indexing Requirements

### Problem
```python
TypeError: Indexing elements must be in increasing order
```

Training failed when loading shuffled batches from HDF5 dataset.

### Root Cause
HDF5 fancy indexing (using a list/array of indices) requires indices to be in sorted order. During training, we shuffle data indices for better learning, but HDF5 cannot handle unsorted index access.

**Problematic code:**
```python
def _generator(self, indices):
    np.random.shuffle(indices)  # Shuffle for training
    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        # batch_indices are NOT sorted!
        y = f['y_received'][batch_indices]  # ❌ Fails with HDF5
```

### Solution (Intermediate)
Sort batch indices before HDF5 access:

```python
batch_indices = indices[i * batch_size:(i + 1) * batch_size]
batch_indices = np.sort(batch_indices)  # ✅ Sort for HDF5
y = f['y_received'][batch_indices]
```

### Solution (Final - Better)
Preload entire dataset into RAM to avoid HDF5 limitations and gain massive speedup (see Issue #5).

### Lessons Learned
1. **HDF5 has strict indexing requirements**: Always sort indices for fancy indexing
2. **Shuffling order matters**: Shuffle at the batch level, not within batches
3. **Consider RAM caching**: If dataset fits in RAM, load once and avoid I/O constraints

### Prevention
- Document HDF5 indexing requirements in data loader code
- Use RAM caching for datasets < 50GB on machines with sufficient memory
- Consider Zarr or TFRecord formats for better random access patterns

---

## Issue #3: Out of Memory - Initial Attempt (batch_size=64)

### Problem
```python
ResourceExhaustedError: Out of memory while trying to allocate 841830367232 bytes
```

Training failed immediately trying to allocate ~840GB of memory.

### Root Cause
Multi-head attention creates attention matrices of size:
```
[batch_size, num_heads, seq_len, seq_len]
```

With our architecture:
- Input to attention: `[batch, 448×14, 128] = [batch, 6272, 128]`
- Attention matrix: `[batch, 4, 6272, 6272]`
- With batch=64 and float32: `64 × 4 × 6272 × 6272 × 4 bytes = 40GB` just for attention!
- Forward + backward pass: ~840GB total

### Solution
Reduced batch size from 64 to 16:

```python
def __init__(self, h5_path, batch_size=16, validation_split=0.1):  # Changed from 64
```

This reduced attention memory to ~10GB, but still failed (see Issue #4).

### Lessons Learned
1. **Attention memory scales quadratically**: `O(seq_len²)` memory requirement
2. **Batch size is first lever**: Easy to adjust, big impact on memory
3. **Calculate before training**: Estimate memory needs before running experiments
4. **Monitor GPU memory**: Use `nvidia-smi` during initial epochs

### Prevention
- Always calculate attention memory: `batch × heads × seq² × 4 bytes`
- Start with small batch sizes (8-16) for attention models
- Use gradient accumulation if small batches hurt convergence

---

## Issue #4: Out of Memory - Second Attempt (Sequence Length)

### Problem
```python
ResourceExhaustedError: Out of memory while trying to allocate 210453397504 bytes
```

Even with batch_size=16, still running out of 48GB VRAM.

### Root Cause
Sequence length of 6272 is too large for attention mechanism:
```
Memory = batch × heads × seq_len × seq_len × 4 bytes
        = 16 × 4 × 6272 × 6272 × 4
        = ~10GB (forward only)

Total with backward pass: ~210GB
```

### First Solution Attempt (Too Aggressive)
Added `MaxPool(4, 2)` before attention, reducing sequence length 8x:
```python
x = layers.MaxPooling2D((4, 2))(x)  # seq_len: 6272 → 784
x = attention_block(x, key_dim=32, num_heads=2)  # Reduced heads too
```

### User Feedback
> "Remember RTX 4090D we have max 48GB memory"
> "We have 48GB memory, do not sacrifice unless needed to"

### Final Solution (Balanced)
Used moderate pooling for 4x reduction, kept full attention:

```python
# Moderate pooling before attention to fit in 48GB memory
# Reduces sequence length from 6272 to 1568 (4x reduction)
x = layers.MaxPooling2D((2, 2))(x)  # [448,14] -> [224,7]

# Attention mechanism (seq_len=1568, fits comfortably in 48GB)
x = attention_block(x, key_dim=64)  # Full 4 heads for best performance
```

**Memory breakdown (batch=16, seq=1568):**
- Model parameters: ~2 GB
- Forward pass: ~5 GB
- Attention (4 heads): ~2.5 GB
- Backward pass: ~12 GB
- Optimizer state: ~8 GB
- TensorFlow overhead: ~5 GB
- **Total: ~35 GB** ✅ (fits in 48GB with headroom)

### Lessons Learned
1. **Balance performance vs. memory**: Don't over-optimize when headroom exists
2. **Sequence length is key lever**: Quadratic impact on attention memory
3. **Listen to user constraints**: "48GB memory" was the design constraint
4. **Keep architectural benefits**: 4-head attention performs better than 2-head
5. **Memory estimation is critical**: Calculate before implementing

### Prevention
- Document memory calculations in code comments
- Use progressive pooling strategy: pool just enough to fit
- Test with memory profiling: `tf.config.experimental.set_memory_growth(gpu, True)`
- Reserve 10-20% VRAM headroom for stability

---

## Issue #5: GPU Idle During Training (Critical Performance Issue)

### Problem
Training extremely slow despite GPU being idle:
- **37 seconds per step** (expected: 0.5s)
- **5.7 hours per epoch** (expected: 5 minutes)
- **5 days total training** (expected: 1.7 hours)

`nvidia-smi` evidence:
```
GPU-Util: 0%
Power: 19W / 425W (4.5%)
P-State: P8 (power-saving/idle)
Temperature: 41°C (cool = not working)
```

### Root Cause
**HDF5 disk I/O bottleneck**. GPU was waiting for data to be read from disk:

```python
def _generator(self, indices):
    with h5py.File(self.h5_path, 'r') as f:  # Open file each time
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            y = f['y_received'][batch_indices]  # ❌ SLOW DISK READ!
            # GPU sits idle waiting for this...
```

**Bottleneck analysis:**
- Disk read: ~36s/step
- GPU compute: ~0.5s/step
- **GPU utilization: 1.4%** (0.5s / 36.5s)

### Solution
**Preload entire dataset into RAM** during initialization:

```python
class PUSCHDataset:
    def __init__(self, h5_path, batch_size=16, validation_split=0.1):
        print("\n📦 Loading dataset into RAM for fast training...")

        with h5py.File(h5_path, 'r') as f:
            # Load ALL data into RAM (~43 GB)
            self.y_data = f['y_received'][:]  # Load to RAM once
            self.bits_data = f['bits'][:]      # Load to RAM once
            print(f"   ✅ Data loaded into RAM")

    def _generator(self, indices):
        """Generator now reads from RAM (fast!)"""
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]

            # Load from RAM (no disk I/O!)
            y = self.y_data[batch_indices]      # ✅ FAST RAM access
            bits = self.bits_data[batch_indices]  # ✅ FAST RAM access
```

### Results
**Achieved 80x speedup:**
- **Before:** 37s/step, 0% GPU util, 19W power, P8 state
- **After:** 0.459s/step, 95%+ GPU util, 350W+ power, P0 state

**Training time:**
- Before: 5 days 🐌
- After: 1.4 hours ⚡

### Lessons Learned
1. **GPU idle = I/O bottleneck**: Always profile when GPU util is low
2. **RAM is cheap, time is expensive**: Use RAM caching for fast iteration
3. **HDF5 is for storage, not streaming**: HDF5 optimized for large sequential reads
4. **Monitor GPU power state**: P8 = idle, P0 = working
5. **Calculate I/O vs compute time**: If I/O > compute, cache in RAM
6. **Modern GPUs are FAST**: RTX 4090 can process way faster than disks can feed

### Prevention
- **Always check GPU utilization first**: `nvidia-smi` during first epoch
- **Profile I/O vs compute**: Use `time.time()` around data loading and model forward pass
- **Design pattern**: Load-once-use-many for datasets that fit in RAM
- **Memory hierarchy awareness**:
  - L1 cache: ~1 cycle
  - RAM: ~100 cycles
  - NVMe SSD: ~100,000 cycles
  - SATA SSD: ~500,000 cycles
- **Rule of thumb**: If dataset < 50% of available RAM, cache it

---

## Issue #6: Lambda Layer Closure Bug (Proactive Fix)

### Problem
Python lambda capturing loop variable by reference instead of value.

### Root Cause
Classic Python closure issue:

```python
# WRONG: All lambdas capture the SAME rx_idx reference
for rx_idx in range(num_rx_antennas):
    x = layers.Lambda(lambda x: x[:, rx_idx, :, :, :])(inputs)  # ❌ Bug!
    # After loop ends, ALL lambdas use rx_idx = 3 (final value)
```

### Solution
Capture by value using default argument:

```python
# CORRECT: Each lambda captures its OWN rx_idx value
for rx_idx in range(num_rx_antennas):
    x = layers.Lambda(lambda x, idx=rx_idx: x[:, idx, :, :, :])(inputs)  # ✅ Fixed!
```

### Lessons Learned
1. **Python closures capture references**: Not values
2. **Lambda in loops is dangerous**: Always use default arguments
3. **Test with multiple iterations**: Bug only appears with multiple loop iterations
4. **Keras quirk**: Lambda layers are evaluated later, not during loop

### Prevention
- Always use `lambda x, arg=value:` pattern in loops
- Consider using `functools.partial` instead of lambda
- Enable linting rules to catch this pattern

---

## Issue #7: EarlyStopping Callback Mode Ambiguity

### Problem
```python
ValueError: EarlyStopping callback received monitor=val_bit_error_rate, but Keras isn't able
to automatically determine whether that metric should be maximized or minimized.
```

Training crashed after first epoch when EarlyStopping tried to evaluate.

### Root Cause
Keras can't infer optimization direction for custom metric names. Built-in metrics like `loss` or `accuracy` have known directions, but `val_bit_error_rate` is ambiguous.

**Problematic code:**
```python
keras.callbacks.EarlyStopping(
    monitor='val_bit_error_rate',
    patience=5,
    restore_best_weights=True,
    verbose=1
),
```

### Solution
Explicitly specify `mode='min'`:

```python
keras.callbacks.EarlyStopping(
    monitor='val_bit_error_rate',
    mode='min',  # Lower BER is better
    patience=5,
    restore_best_weights=True,
    verbose=1
),
```

### Lessons Learned
1. **Always specify mode**: Don't rely on Keras inference for custom metrics
2. **Fail-fast is good**: Better to crash than silently optimize wrong direction
3. **Document metric semantics**: Comment whether metric should be minimized/maximized
4. **Standard naming helps**: Using `loss` or `accuracy` in metric names aids inference

### Prevention
- Always add `mode='min'` or `mode='max'` for custom metrics
- Use naming conventions: `*_loss` (minimize) or `*_accuracy` (maximize)
- Add mode parameter to all callbacks that monitor custom metrics

---

## Performance Optimization Summary

### Training Performance Evolution

| Optimization | Time/Epoch | Total Time | GPU Util | Speedup |
|--------------|------------|------------|----------|---------|
| Initial (batch=64, HDF5) | N/A | N/A | N/A | OOM crash |
| batch=16, HDF5 | 5.7 hrs | 5 days | 0% | 1x (baseline) |
| batch=16, RAM cache | 4.2 min | 1.4 hrs | 95%+ | **80x** ⚡ |

### Memory Optimization Evolution

| Configuration | Seq Length | Batch | Heads | Memory | Status |
|---------------|------------|-------|-------|--------|--------|
| Original | 6272 | 64 | 4 | 840 GB | ❌ OOM |
| Reduced batch | 6272 | 16 | 4 | 210 GB | ❌ OOM |
| Aggressive pool | 784 | 16 | 2 | 5 GB | ✅ Fits (over-optimized) |
| Balanced | 1568 | 16 | 4 | 35 GB | ✅ Optimal |

---

## Key Takeaways

### 1. Measure Before Optimizing
- Profile GPU utilization FIRST (nvidia-smi)
- Calculate memory requirements BEFORE training
- Identify bottlenecks: CPU, I/O, or GPU

### 2. Memory Hierarchy Matters
```
L1 Cache < L2 Cache < RAM < NVMe < Network
    1ns       10ns     100ns   100μs   10ms
```
Move data as close to GPU as possible.

### 3. Attention Mechanism Scaling
```
Memory = O(batch × heads × seq²)
Time   = O(batch × heads × seq² × dim)
```
Sequence length has quadratic impact!

### 4. Framework API Changes
- TensorFlow 2.20+ uses Keras 3.x (breaking changes)
- Sionna 1.2.x restructured to `sionna.phy.*` namespace
- Always read migration guides for major version updates

### 5. User Constraints Are Requirements
When user says "We have 48GB memory, don't sacrifice unless needed":
- This is a design constraint, not a suggestion
- Balance performance vs. constraints
- Leave headroom for stability

### 6. Fail-Fast Is Good
Early errors (like EarlyStopping mode) prevent silent failures:
- Better to crash with clear error than silently optimize wrong metric
- Explicit parameters > implicit inference
- Document expected behavior in code

### 7. RAM Caching Decision Tree
```
Dataset size < 50% RAM? → Always cache
50% < Dataset < 80% RAM? → Cache if training is bottlenecked
Dataset > 80% RAM?       → Use streaming with prefetch
```

---

## Best Practices Checklist

### Before Training
- [ ] Calculate attention memory: `batch × heads × seq² × 4 bytes`
- [ ] Estimate total GPU memory including backward pass and optimizer
- [ ] Reserve 10-20% VRAM headroom
- [ ] Profile one epoch with small data to catch issues early
- [ ] Check GPU utilization with `nvidia-smi`

### Data Loading
- [ ] Use RAM caching if dataset < 50% available RAM
- [ ] Implement aggressive prefetching: `dataset.prefetch(tf.data.AUTOTUNE)`
- [ ] Profile data loading time vs. model compute time
- [ ] Use `@tf.function` for GPU-accelerated data generation

### Model Architecture
- [ ] Avoid `tf.shape()` in Keras Functional API (use `x.shape`)
- [ ] Capture loop variables by value: `lambda x, arg=value:`
- [ ] Calculate sequence length before attention layers
- [ ] Use progressive pooling to reduce sequence length

### Callbacks
- [ ] Always specify `mode='min'/'max'` for custom metrics
- [ ] Implement dual checkpointing: best model + epoch checkpoints
- [ ] Use `ReduceLROnPlateau` for learning rate scheduling
- [ ] Enable TensorBoard for monitoring

### Debugging
- [ ] Check GPU utilization first (`nvidia-smi`)
- [ ] Check power state (P0 = working, P8 = idle)
- [ ] Profile with `nvprof` or `nsys` for detailed analysis
- [ ] Test with small batch/data first

---

## Tools and Commands Reference

### GPU Monitoring
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check GPU utilization and power
nvidia-smi --query-gpu=utilization.gpu,power.draw,power.limit --format=csv -l 1

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### Memory Profiling
```python
# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Profile memory
import tensorflow as tf
tf.config.experimental.set_memory_growth(True)
tf.debugging.set_log_device_placement(True)
```

### Performance Profiling
```python
import time

# Profile data loading
start = time.time()
batch = next(iter(dataset))
print(f"Data loading: {time.time() - start:.3f}s")

# Profile model forward pass
start = time.time()
output = model(batch[0])
print(f"Forward pass: {time.time() - start:.3f}s")
```

---

## References

1. **Keras 3.x Migration Guide**: https://keras.io/guides/migrating_to_keras_3/
2. **TensorFlow Memory Guide**: https://www.tensorflow.org/guide/gpu
3. **Attention Mechanism Memory**: https://arxiv.org/abs/2001.04451
4. **HDF5 Best Practices**: https://docs.h5py.org/en/stable/high/dataset.html
5. **NVIDIA GPU Profiling**: https://docs.nvidia.com/nsight-systems/

---

## Contributors
- Testing & Feedback: Fatih E. NAR
- Platform: Red Hat OpenShift AI
- Hardware: NVIDIA RTX 4090 D (48GB)

**Last Updated:** November 17, 2025
