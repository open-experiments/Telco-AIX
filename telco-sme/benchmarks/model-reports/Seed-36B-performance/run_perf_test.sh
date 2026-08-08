#!/bin/bash

# Performance Test Runner for Seed-OSS-36B Deployments
# Usage: ./run_perf_test.sh [quick|full|compare]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
KUBECONFIG_PATH="/Users/fenar/projects/clusters/sandbox02/kubeconfigX"
NAMESPACE="tme-aix"

echo -e "${GREEN}=== Seed-OSS-36B Performance Test Runner ===${NC}"
echo ""

# Get API key from OpenShift secret
if [ -f "$KUBECONFIG_PATH" ]; then
    export KUBECONFIG=$KUBECONFIG_PATH
    echo -e "${GREEN}✓${NC} Using kubeconfig: $KUBECONFIG_PATH"
else
    echo -e "${RED}✗${NC} Kubeconfig not found at $KUBECONFIG_PATH"
    exit 1
fi

# Check if we're logged in
if ! oc whoami >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} Not logged into OpenShift. Please login first."
    exit 1
fi

# Get API key
API_KEY=$(oc get secret seed-oss-api-key -n $NAMESPACE -o jsonpath='{.data.api-key}' | base64 -d)
if [ -z "$API_KEY" ]; then
    echo -e "${RED}✗${NC} Failed to retrieve API key from secret"
    exit 1
fi

echo -e "${GREEN}✓${NC} Retrieved API key from OpenShift secret"

# Check Python dependencies
echo -e "${YELLOW}Checking Python dependencies...${NC}"
python3 -c "import aiohttp, asyncio" 2>/dev/null || {
    echo -e "${YELLOW}Installing required Python packages...${NC}"
    pip3 install aiohttp asyncio
}

# Determine test mode
MODE=${1:-"compare"}
RUNS=${2:-3}

case $MODE in
    "quick")
        echo -e "${BLUE}Running quick test (1 run each endpoint)...${NC}"
        RUNS=1
        python3 perf_test_suite.py --api-key "$API_KEY" --runs $RUNS --endpoint both --output "perf_report_quick_$(date +%Y%m%d_%H%M%S).md"
        ;;
    "full")
        echo -e "${BLUE}Running full test (5 runs each endpoint)...${NC}"
        RUNS=5
        python3 perf_test_suite.py --api-key "$API_KEY" --runs $RUNS --endpoint both --output "perf_report_full_$(date +%Y%m%d_%H%M%S).md"
        ;;
    "compare")
        echo -e "${BLUE}Running comparison test ($RUNS runs each endpoint)...${NC}"
        python3 perf_test_suite.py --api-key "$API_KEY" --runs $RUNS --endpoint both --output "perf_report_compare_$(date +%Y%m%d_%H%M%S).md"
        ;;
    "standard")
        echo -e "${BLUE}Testing standard endpoint only...${NC}"
        python3 perf_test_suite.py --api-key "$API_KEY" --runs $RUNS --endpoint standard
        ;;
    "hp")
        echo -e "${BLUE}Testing high-performance endpoint only...${NC}"
        python3 perf_test_suite.py --api-key "$API_KEY" --runs $RUNS --endpoint hp
        ;;
    *)
        echo -e "${RED}Usage: $0 [quick|full|compare|standard|hp] [runs]${NC}"
        echo ""
        echo "Test modes:"
        echo "  quick    - 1 run each endpoint, fast results"
        echo "  full     - 5 runs each endpoint, comprehensive"
        echo "  compare  - 3 runs each endpoint, balanced (default)"
        echo "  standard - Test standard endpoint only"
        echo "  hp       - Test high-performance endpoint only"
        echo ""
        echo "Optional: Number of runs per endpoint (default: 3)"
        echo ""
        echo "Examples:"
        echo "  $0 quick          # Quick comparison test"
        echo "  $0 full           # Full comprehensive test"
        echo "  $0 compare 2      # Compare with 2 runs each"
        echo "  $0 standard 1     # Test standard endpoint once"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Performance testing complete!${NC}"
echo -e "${YELLOW}Check the generated report file for detailed results.${NC}"

# Show recent performance metrics from deployments
echo ""
echo -e "${BLUE}Recent deployment metrics:${NC}"
echo "Standard deployment:"
oc logs deployment/seed-oss-36b-vllm -n $NAMESPACE --tail=2 | grep -E "(metrics|throughput)" | tail -1 || echo "  No recent metrics"

echo "High-performance deployment:"
oc logs deployment/seed-oss-36b-vllm-hp -n $NAMESPACE --tail=2 | grep -E "(metrics|throughput)" | tail -1 || echo "  No recent metrics"

echo ""
echo -e "${YELLOW}Tip: Run with different modes to compare results:${NC}"
echo "  ./run_perf_test.sh quick    # For rapid testing"
echo "  ./run_perf_test.sh full     # For detailed analysis"