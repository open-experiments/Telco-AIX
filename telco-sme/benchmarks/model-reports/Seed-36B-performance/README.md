# Performance Testing Suite

Comprehensive performance testing suite for Seed-OSS-36B optimized deployment.

## Files

- **`perf_test_suite.py`**: Main testing engine with streaming support
- **`run_perf_test.sh`**: Easy-to-use test runner
- **`requirements.txt`**: Python dependencies

## Test Scenarios

8 telecom-specific test cases covering:
1. **5G Network Optimization** (300 tokens)
2. **Network Troubleshooting** (250 tokens)
3. **Edge Computing Analysis** (400 tokens)
4. **Network Slicing Configuration** (350 tokens)
5. **Spectrum Management** (200 tokens)
6. **Customer Service Response** (150 tokens)
7. **Network Security Assessment** (300 tokens)
8. **Infrastructure Planning** (250 tokens)

## Metrics Measured

- **Tokens per second** (throughput)
- **Time to first token** (latency)
- **Inter-token latency** (streaming performance)
- **Total response time**
- **Success rate** & error handling

## Usage

### Quick Test (1 run each endpoint)
```bash
./run_perf_test.sh quick
```

### Full Test (5 runs each endpoint)
```bash
./run_perf_test.sh full
```

### Performance Test (3 runs - default)
```bash
./run_perf_test.sh compare
```

### Single Endpoint Test
```bash
./run_perf_test.sh optimized   # Test optimized endpoint
```

## Output

- Detailed markdown reports with performance analysis
- Performance metrics and statistics
- Recommendations based on results
- Technical metrics and statistics

## Dependencies

```bash
pip3 install -r requirements.txt
```

## Requirements

- Python 3.7+
- OpenShift access with valid kubeconfig
- Optimized deployment must be running and healthy