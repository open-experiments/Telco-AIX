# Management Scripts

## Scripts

- **`deploy.sh`**: All-in-one deployment, monitoring, and optimization tool
- **`cleanup.sh`**: Complete cleanup with confirmation prompts

## Usage

### Deploy and Management
```bash
./deploy.sh [command]
```

**Available Commands:**
- `deploy` - Deploy new instance with API key generation
- `status` - Show current status and performance metrics  
- `monitor` - Live performance monitoring (streaming)
- `restart` - Rolling restart for optimization
- `scale` - Scale deployment replicas (0-1)
- `gpu` - Check GPU utilization with nvidia-smi
- `config` - View deployment configuration details
- `perf` - Run performance test
- `test` - Test API endpoint with sample request

**Interactive Mode:** Run `./deploy.sh` without arguments for menu

### Complete Cleanup
```bash
./cleanup.sh
```
Options:
- Remove all deployment resources
- Preserve secrets for redeployment 
- Confirmation prompts and verification

## Requirements

- OpenShift CLI (`oc`) installed
- Access to `tme-aix` namespace
- Kubeconfig: `/Users/fenar/projects/clusters/sandbox02/kubeconfigX`