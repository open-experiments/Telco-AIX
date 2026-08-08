# Deployment Manifests

## Files

- **`seed-oss-36b-deployment.yaml`**: Optimized deployment configuration

## Usage

```bash
# Deploy optimized configuration
oc apply -f seed-oss-36b-deployment.yaml
```

## Optimized Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Model Length | 32,768 | Maximum context window |
| Max Sequences | 1,024 | Concurrent request handling |
| GPU Memory Utilization | 95% | Maximum GPU usage |
| Memory Request/Limit | 90Gi/96Gi | Pod memory allocation |
| Shared Memory | 32Gi | High-performance IPC |
| Expected Performance | 25-35 tokens/s | Target throughput |
| Context Window | 32K tokens | Full model capability |

## Features

- **Optimized for RTX PRO 6000 96GB GPU**
- **vLLM v0.10.1.1 runtime** 
- **Chunked prefill enabled**
- **Prefix caching enabled**
- **OpenShift security compliant**