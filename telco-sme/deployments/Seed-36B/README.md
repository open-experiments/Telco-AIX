# Seed-OSS-36B vLLM Deployment Project

This repository contains the complete deployment and performance testing suite for running ByteDance Seed-OSS-36B-Instruct model on Red Hat OpenShift with vLLM runtime.

## Project Structure

```
├── deployments/           # Kubernetes deployment manifests
├── scripts/              # Management and deployment scripts  
├── performance/          # Performance testing suite
├── docs/                 # Documentation and reports
└── README.md            # This file
```

## Quick Start

### Deploy and Manage
```bash
cd scripts && ./deploy.sh deploy     # Deploy new instance
cd scripts && ./deploy.sh status     # Show current status  
cd scripts && ./deploy.sh monitor    # Live performance monitoring
cd scripts && ./deploy.sh test       # Test API endpoint
```

### Run Performance Tests  
```bash
cd performance/
./run_perf_test.sh compare
```

### Complete Cleanup
```bash
cd scripts && ./cleanup.sh
```

## Current Deployment Status
- **API Endpoint**: `https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io`
- **Model**: ByteDance-Seed/Seed-OSS-36B-Instruct (72GB)
- **Hardware**: Nvidia RTX PRO 6000 96GB vRAM Blackwell GPU
- **Configuration**: Optimized for maximum performance (32K context)

## Key Features

✅ **Optimized Deployment**: 32K context window, 95% GPU utilization  
✅ **OpenShift Optimized**: Security constraints and resource management  
✅ **Performance Testing**: 8 telecom-specific test scenarios  
✅ **Automated Scripts**: Deploy, cleanup, and monitoring tools  
✅ **Comprehensive Monitoring**: Metrics collection and analysis  

See individual folders for detailed documentation.