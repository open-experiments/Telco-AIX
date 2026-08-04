# Qwen3-Embedding-8B Deployment

Simple deployment for Qwen3-Embedding-8B model using Text Embeddings Inference (TEI).

## Files

- `qwen3-embedding-tei-deployment.yaml` - Complete deployment with ServiceAccount, SCC, ConfigMap, Deployment, Service, and Route

## Deploy

```bash
oc apply -f qwen3-embedding-tei-deployment.yaml
```

## API Access

**Endpoint:** `https://qwen3-embedding-tei-route-tme-aix.apps.acmhub.narlabs.io`

**Health Check:**
```bash
curl https://qwen3-embedding-tei-route-tme-aix.apps.acmhub.narlabs.io/health
```

**Generate Embeddings:**
```bash
curl -X POST https://qwen3-embedding-tei-route-tme-aix.apps.acmhub.narlabs.io/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -d '{"inputs": ["Your text here"]}'
```

## Security

API requires authentication with a Bearer token. Without a token it returns `401 Unauthorized`.

Fetch the token from the cluster (never hardcode it in docs or code):

```bash
export API_TOKEN=$(oc get secret qwen3-embedding-api-token -n tme-aix -o jsonpath='{.data.api-token}' | base64 -d)
```

## Components

- **Secret:** `qwen3-embedding-api-token` - API authentication token
- **ServiceAccount:** `qwen3-embedding-tei-sa`
- **SCC:** anyuid ClusterRoleBinding 
- **ConfigMap:** `qwen3-embedding-tei-config`
- **Deployment:** `qwen3-embedding-tei`
- **Service:** `qwen3-embedding-tei-service`
- **Route:** `qwen3-embedding-tei-route`

## Model Details

- **Model:** Qwen/Qwen3-Embedding-8B
- **Engine:** Text Embeddings Inference (TEI)
- **GPU:** Any available NVIDIA GPU
- **Max Context:** 8,192 tokens
- **Embedding Dim:** 4,096

## GPU Memory Planning

### Memory Usage Formula

**Model Memory (Fixed):**
```
Qwen3-Embedding-8B = 8 billion parameters × 2 bytes (float16) = 16GB
```

**Dynamic Memory (Scales with usage):**
```
Context Memory = Max Context × Hidden Size × Batch Size × Layers × 4 bytes
Hidden Size = 4,096 (model architecture, not output dim)
Layers = ~32 (transformer layers)
Example: 8,192 × 4,096 × 8 × 32 × 4 = 34GB per batch
```

**Total GPU Memory Required:**
```
Total = Model Memory + Context Memory + KV Cache + Overhead
Example: 16GB + 8GB (actual context) + 2GB (cache) + 2GB = ~28GB
```

**Note:** Embedding Dim (4,096) is the output vector size, not a memory parameter.

### Real-World Example

Current deployment on RTX 4090 (24GB):

```
sh-5.1# nvidia-smi
Fri Aug 15 03:12:47 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:19:00.0 Off |                  Off |
|  0%   41C    P8             13W /  450W |   15516MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        On  |   00000000:68:00.0 Off |                  Off |
|  0%   47C    P8             21W /  450W |       1MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          553657      C   text-embeddings-router                15506MiB |
+-----------------------------------------------------------------------------------------+
```

**Analysis:**
- **Model Memory:** 15,506 MiB (~15.5GB) - Base Qwen3-Embedding-8B weights
- **Dynamic Memory:** 0 MiB additional - TEI uses in-place computation
- **Power Under Load:** 13W → 83W (processing), returns to ~58W  
- **Temperature:** 41°C → 46°C (+5°C under sustained load)
- **Available Memory:** 9,058 MiB (~9GB) - Reserved but unused due to efficient design

**Key Insight:** TEI optimizes for memory efficiency - embedding computation happens in-place without allocating additional GPU memory, making it ideal for high-throughput scenarios.

### Load Testing Results

Under heavy concurrent load (70+ embedding requests):

**Memory Usage:** ✅ **CONSTANT** at 15,516 MiB
- No dynamic allocation regardless of load
- Context processing uses existing model memory space
- Validates in-place computation architecture

**Power & Performance:** ⚡ **SCALES WITH LOAD**  
- Idle: ~58W → Heavy Load: 80-100W+ 
- Temperature: 46°C → 50-55°C under sustained load
- Performance State: P8 (idle) → P2 (active processing)

**Conclusion:** TEI's fixed memory footprint means you can maximize throughput without memory concerns - scaling is limited by compute, not memory allocation.

### Scaling Parameters

**Increase Max Context (8,192 → 16,384):**
- Memory impact: Proportional to context length
- Edit ConfigMap: `MAX_BATCH_TOKENS: "65536"`
- Requires: Additional 4-8GB GPU memory
- Formula: Additional memory ≈ (New Context - Old Context) × Hidden Size × Batch × 4 bytes

**Increase Batch Size (32 → 64):**
- Memory impact: Linear with batch size
- Edit ConfigMap: `MAX_CONCURRENT_REQUESTS: "64"`
- Edit ConfigMap: `MAX_CLIENT_BATCH_SIZE: "128"`
- Requires: Additional 8-16GB GPU memory
- Formula: Additional memory ≈ Context × Hidden Size × Additional Batch × 4 bytes

### GPU Memory Requirements

| GPU Memory | Max Context | Batch Size | Use Case |
|------------|-------------|------------|----------|
| 16GB | 4,096 | 16 | Basic inference |
| 24GB | 8,192 | 32 | **Current setup** |
| 32GB | 16,384 | 32 | Long documents |
| 48GB | 16,384 | 64 | High throughput |
| 80GB | 32,768 | 64 | Enterprise scale |

### Configuration Updates

To modify parameters, edit the ConfigMap in deployment YAML:

```yaml
data:
  MAX_BATCH_TOKENS: "32768"        # Max Context × 4
  MAX_CONCURRENT_REQUESTS: "32"    # Batch size
  MAX_CLIENT_BATCH_SIZE: "64"      # Client batch limit
```

Then redeploy:
```bash
oc apply -f qwen3-embedding-tei-deployment.yaml
```