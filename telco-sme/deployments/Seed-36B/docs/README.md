# Documentation

This folder contains project documentation and reports.

## Files

- **`performance_report_20250828.md`**: Comprehensive performance analysis comparing deployment configurations

## Key Documentation Topics

### Deployment Guide
See the main [README.md](../README.md) for quick start instructions.

### Performance Analysis
The performance report shows detailed analysis of the optimized deployment:
- **Performance**: ~20 tokens/s generation throughput
- **Context**: 32K tokens maximum
- **GPU Utilization**: 95% optimized usage
- **Memory**: 90-96GB allocated for maximum performance

### Architecture Notes

The deployment overcame several technical challenges:

1. **Seed-OSS Architecture Support**: Required transformers installation from source via init container
2. **OpenShift Security Constraints**: Used VLLM_USE_V1=0 to avoid v1 engine permission issues  
3. **Cache Directory Permissions**: Redirected all cache directories to /tmp for OpenShift compatibility
4. **GPU Resource Management**: Optimized for single RTX PRO 6000 96GB GPU with 95% utilization

### Current Status

**API Endpoint**: `https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io`  
**Model**: ByteDance-Seed/Seed-OSS-36B-Instruct (72GB)  
**Configuration**: Optimized for maximum performance  
**Expected Performance**: 25-35 tokens/s with 32K context capability

## API Usage Examples

### Test the API
```bash
# Get API key from OpenShift secret
API_KEY=$(oc get secret seed-oss-api-key -n tme-aix -o jsonpath='{.data.api-key}' | base64 -d)

# Simple test
curl -k -X POST "https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    --data '{
        "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "messages": [{"role": "user", "content": "What is 5G network slicing?"}],
        "temperature": 0.7,
        "max_tokens": 100
    }'
```

### Monitoring
```bash
# Health check
curl -k https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io/health

# Check deployment logs
oc logs -f deployment/seed-oss-36b-vllm -n tme-aix | grep metrics
```

## Web Application Integration

Configure your web application to use the API:

```python
config = Config()
config.api_endpoint = "https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io"
config.model_name = "ByteDance-Seed/Seed-OSS-36B-Instruct"
config.api_token = "YOUR_API_KEY_HERE"  # Use the generated API key
config.use_token_auth = True
```

## License

This deployment configuration is provided as-is. The Seed-OSS-36B model has its own license terms on HuggingFace.