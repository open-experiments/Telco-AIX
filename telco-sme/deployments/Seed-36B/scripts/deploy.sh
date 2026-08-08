#!/bin/bash

# Seed-OSS-36B vLLM Deployment Script - Deploy, Monitor, and Optimize
# Author: Fatih E NAR
# Date: 2025

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="tme-aix"
KUBECONFIG_PATH="/Users/fenar/projects/clusters/sandbox02/kubeconfigX"

echo -e "${GREEN}=== Seed-OSS-36B vLLM Deployment & Management Script ===${NC}"
echo ""

# Check if running with correct kubeconfig
if [ -f "$KUBECONFIG_PATH" ]; then
    export KUBECONFIG=$KUBECONFIG_PATH
    echo -e "${GREEN}✓${NC} Using kubeconfig: $KUBECONFIG_PATH"
else
    echo -e "${YELLOW}⚠${NC} Kubeconfig not found at $KUBECONFIG_PATH"
    echo "Using current kubeconfig: $KUBECONFIG"
fi

# Check current context
echo -e "\n${YELLOW}Current OpenShift context:${NC}"
oc whoami -c || { echo -e "${RED}✗${NC} Failed to get current context. Please login to OpenShift first."; exit 1; }

# Check namespace
oc project $NAMESPACE >/dev/null 2>&1 || { 
    echo -e "${YELLOW}⚠${NC} Namespace $NAMESPACE not found."
    if [[ "$1" == "deploy" ]]; then
        echo "Creating namespace..."
        oc new-project $NAMESPACE
    else
        exit 1
    fi
}

echo -e "${BLUE}Deployment & Management Options:${NC}"
echo "1) Deploy new instance"
echo "2) Show current status and metrics"
echo "3) Monitor live performance (streaming)"
echo "4) Restart deployment for optimization"
echo "5) Scale deployment replicas"
echo "6) Check GPU utilization"
echo "7) View deployment configuration"
echo "8) Run performance test"
echo "9) Test API endpoint"
echo ""

# Auto-select based on command line argument or prompt user
if [ "$#" -eq 1 ]; then
    case $1 in
        "deploy"|"1") choice=1 ;;
        "status"|"2") choice=2 ;;
        "monitor"|"3") choice=3 ;;
        "restart"|"4") choice=4 ;;
        "scale"|"5") choice=5 ;;
        "gpu"|"6") choice=6 ;;
        "config"|"7") choice=7 ;;
        "perf"|"8") choice=8 ;;
        "test"|"9") choice=9 ;;
        *) echo -e "${RED}Invalid argument. Use: deploy|status|monitor|restart|scale|gpu|config|perf|test${NC}"; exit 1 ;;
    esac
else
    read -p "Select option (1-9): " choice
fi

case $choice in
    1)
        echo -e "\n${YELLOW}=== DEPLOYING SEED-OSS-36B ===${NC}"
        
        # Check if deployment already exists
        if oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠${NC} Deployment already exists. Use option 4 to restart or cleanup first."
            exit 1
        fi
        
        # Generate API key
        echo -e "\n${YELLOW}Generating API authentication key...${NC}"
        API_KEY="sk-seed-oss-36b-$(openssl rand -hex 32)"
        echo -e "${GREEN}✓${NC} API Key generated (save this for client access):"
        echo -e "${GREEN}$API_KEY${NC}"
        echo ""
        
        # Create secrets
        echo -e "${YELLOW}Creating secrets...${NC}"
        cat <<EOF | oc apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: seed-oss-api-key
  namespace: $NAMESPACE
  labels:
    app: seed-oss-36b
type: Opaque
stringData:
  api-key: "$API_KEY"
EOF

        cat <<EOF | oc apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: seed-oss-hf-token
  namespace: $NAMESPACE
  labels:
    app: seed-oss-36b
type: Opaque
stringData:
  token: ""
EOF
        echo -e "${GREEN}✓${NC} Secrets created"
        
        # Deploy resources
        echo -e "\n${YELLOW}Deploying all resources...${NC}"
        oc apply -f ../deployments/seed-oss-36b-deployment.yaml
        echo -e "${GREEN}✓${NC} All resources deployed"
        
        # Wait for deployment
        echo -e "\n${YELLOW}Waiting for deployment to be ready...${NC}"
        echo "This may take 10-15 minutes as the model needs to be downloaded (~72GB)"
        oc rollout status deployment/seed-oss-36b-vllm -n $NAMESPACE --timeout=900s || {
            echo -e "${YELLOW}⚠${NC} Deployment is taking longer than expected. Check logs with:"
            echo "oc logs -f deployment/seed-oss-36b-vllm -n $NAMESPACE"
        }
        
        # Get Route URL
        ROUTE_URL=$(oc get route seed-oss-36b-route -n $NAMESPACE -o jsonpath='{.spec.host}')
        echo -e "\n${GREEN}=== Deployment Complete ===${NC}"
        echo -e "API Endpoint: ${GREEN}https://$ROUTE_URL${NC}"
        echo -e "API Key: ${GREEN}$API_KEY${NC}"
        echo ""
        echo -e "${YELLOW}Test with: ./deploy.sh test${NC}"
        ;;
        
    2)
        echo -e "\n${YELLOW}=== CURRENT STATUS & METRICS ===${NC}"
        
        # Show deployment status
        echo -e "\n${BLUE}Deployment Status:${NC}"
        oc get deployment seed-oss-36b-vllm -o wide 2>/dev/null || echo "Deployment not found"
        
        echo -e "\n${BLUE}Pod Status:${NC}"
        oc get pods -l app=seed-oss-36b 2>/dev/null || echo "No pods found"
        
        echo -e "\n${BLUE}Recent Performance Metrics:${NC}"
        if oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
            echo "Recent logs with metrics:"
            oc logs deployment/seed-oss-36b-vllm --tail=10 | grep -E "(metrics|throughput|Avg)" || echo "No recent metrics found"
        fi
        
        echo -e "\n${BLUE}Route Information:${NC}"
        ROUTE_URL=$(oc get route seed-oss-36b-route -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null)
        if [ -n "$ROUTE_URL" ]; then
            echo "API Endpoint: https://$ROUTE_URL"
            
            # Test health endpoint
            echo -e "\n${BLUE}Health Check:${NC}"
            if curl -k -s -w "HTTP %{http_code}" "https://$ROUTE_URL/health" --max-time 5 >/dev/null 2>&1; then
                echo "✅ API is responding"
            else
                echo "❌ API not responding or still starting"
            fi
        else
            echo "Route not found"
        fi
        ;;
        
    3)
        echo -e "\n${YELLOW}Monitoring live performance (press Ctrl+C to stop)...${NC}"
        
        if oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
            oc logs -f deployment/seed-oss-36b-vllm -n $NAMESPACE | grep --line-buffered -E "(metrics|throughput|Avg|INFO|ERROR)"
        else
            echo -e "${RED}✗${NC} Deployment not found"
        fi
        ;;
        
    4)
        echo -e "\n${YELLOW}=== RESTARTING DEPLOYMENT ===${NC}"
        
        if ! oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
            echo -e "${RED}✗${NC} Deployment not found. Use option 1 to deploy first."
            exit 1
        fi
        
        echo -e "${YELLOW}Rolling restart of deployment...${NC}"
        oc rollout restart deployment/seed-oss-36b-vllm -n $NAMESPACE
        
        echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
        oc rollout status deployment/seed-oss-36b-vllm -n $NAMESPACE --timeout=600s
        
        echo -e "${GREEN}✓${NC} Deployment restarted successfully!"
        ;;
        
    5)
        echo -e "\n${YELLOW}=== SCALING DEPLOYMENT ===${NC}"
        
        if ! oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
            echo -e "${RED}✗${NC} Deployment not found. Use option 1 to deploy first."
            exit 1
        fi
        
        CURRENT_REPLICAS=$(oc get deployment seed-oss-36b-vllm -o jsonpath='{.spec.replicas}')
        echo "Current replicas: $CURRENT_REPLICAS"
        echo ""
        echo -e "${YELLOW}Warning: Each replica needs 96GB GPU memory${NC}"
        echo "Available scaling options:"
        echo "1 - Single replica (recommended for 96GB GPU)"
        echo "0 - Scale down (stops the service)"
        echo ""
        
        read -p "Enter desired replica count (0-1): " replicas
        
        if [[ "$replicas" =~ ^[0-1]$ ]]; then
            oc scale deployment/seed-oss-36b-vllm --replicas=$replicas -n $NAMESPACE
            echo -e "${GREEN}✓${NC} Scaled to $replicas replicas"
            
            if [ "$replicas" -gt 0 ]; then
                echo -e "${YELLOW}Waiting for deployment...${NC}"
                oc rollout status deployment/seed-oss-36b-vllm -n $NAMESPACE --timeout=600s
            fi
        else
            echo -e "${RED}✗${NC} Invalid replica count. Must be 0 or 1."
        fi
        ;;
        
    6)
        echo -e "\n${YELLOW}=== GPU UTILIZATION ===${NC}"
        
        if oc get pods -l app=seed-oss-36b >/dev/null 2>&1; then
            POD_NAME=$(oc get pods -l app=seed-oss-36b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ -n "$POD_NAME" ]; then
                echo "Pod: $POD_NAME"
                echo ""
                echo -e "${BLUE}GPU Information:${NC}"
                oc exec $POD_NAME -- nvidia-smi || echo "Could not access GPU information"
                
                echo -e "\n${BLUE}GPU Memory Usage:${NC}"
                oc exec $POD_NAME -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Could not get memory info"
            else
                echo "No running pods found"
            fi
        else
            echo "No pods with app=seed-oss-36b found"
        fi
        ;;
        
    7)
        echo -e "\n${YELLOW}=== DEPLOYMENT CONFIGURATION ===${NC}"
        
        if oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
            echo -e "\n${BLUE}Deployment Details:${NC}"
            oc describe deployment seed-oss-36b-vllm | head -30
            
            echo -e "\n${BLUE}Resource Limits:${NC}"
            oc get deployment seed-oss-36b-vllm -o jsonpath='{.spec.template.spec.containers[0].resources}' | jq . 2>/dev/null || echo "Could not parse resources"
            
            echo -e "\n${BLUE}Key Environment Variables:${NC}"
            oc get deployment seed-oss-36b-vllm -o jsonpath='{.spec.template.spec.containers[0].env[*]}' | jq '.[] | select(.name | contains("VLLM") or contains("GPU") or contains("CUDA"))' 2>/dev/null || echo "Could not parse environment variables"
        else
            echo -e "${RED}✗${NC} Deployment not found"
        fi
        ;;
        
    8)
        echo -e "\n${YELLOW}=== PERFORMANCE TEST ===${NC}"
        
        # Check if performance test exists
        if [ -f "../performance/quick_perf_test.py" ]; then
            echo "Starting quick performance test..."
            cd ../performance
            export KUBECONFIG=$KUBECONFIG_PATH
            python3 quick_perf_test.py
            cd ../scripts
        else
            echo -e "${RED}✗${NC} Performance test script not found"
            echo "Try: cd ../performance && ./run_perf_test.sh quick"
        fi
        ;;
        
    9)
        echo -e "\n${YELLOW}=== API TEST ===${NC}"
        
        ROUTE_URL=$(oc get route seed-oss-36b-route -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null)
        if [ -z "$ROUTE_URL" ]; then
            echo -e "${RED}✗${NC} Route not found. Deploy first with option 1."
            exit 1
        fi
        
        # Get API key
        API_KEY=$(oc get secret seed-oss-api-key -n $NAMESPACE -o jsonpath='{.data.api-key}' | base64 -d 2>/dev/null)
        if [ -z "$API_KEY" ]; then
            echo -e "${RED}✗${NC} API key not found. Deploy first with option 1."
            exit 1
        fi
        
        echo "Testing API endpoint: https://$ROUTE_URL"
        echo ""
        
        # Health check first
        echo -e "${BLUE}Health Check:${NC}"
        if curl -k -s -w "HTTP %{http_code}\n" "https://$ROUTE_URL/health" --max-time 10; then
            echo ""
            
            # Simple API test
            echo -e "${BLUE}API Test:${NC}"
            curl -k -X POST "https://$ROUTE_URL/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $API_KEY" \
                --data '{
                    "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
                    "messages": [{"role": "user", "content": "Hello! What are you?"}],
                    "temperature": 0.7,
                    "max_tokens": 50
                }' --max-time 60 | jq . 2>/dev/null || echo "Response received (jq not available for formatting)"
        else
            echo -e "\n${RED}✗${NC} Health check failed. API may still be starting up."
        fi
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}Performance Tips:${NC}"
echo "• Monitor GPU KV cache usage in logs"
echo "• Use batch requests for maximum throughput"
echo "• Enable streaming for better perceived performance"
echo "• Current config supports up to 32K token context"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "• Check logs: oc logs -f deployment/seed-oss-36b-vllm -n $NAMESPACE"
echo "• Quick status: ./deploy.sh status"
echo "• Live monitoring: ./deploy.sh monitor"