#!/bin/bash

# Cleanup Seed-OSS-36B-Instruct vLLM deployment from OpenShift
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

echo -e "${RED}=== Seed-OSS-36B vLLM Cleanup Script ===${NC}"
echo -e "${YELLOW}⚠ WARNING: This will remove the entire Seed-OSS-36B deployment${NC}"
echo -e "${BLUE}For deployment and management, use: ./deploy.sh${NC}"
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

# Check if namespace exists
if ! oc get namespace $NAMESPACE >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Namespace $NAMESPACE not found. Nothing to cleanup."
    exit 0
fi

# Switch to namespace
oc project $NAMESPACE >/dev/null 2>&1

# Check if deployment exists
if ! oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Seed-OSS-36B deployment not found in namespace $NAMESPACE"
    echo "Checking for any remaining resources..."
else
    echo -e "${BLUE}Found Seed-OSS-36B deployment in namespace: $NAMESPACE${NC}"
fi

# Show current resources before cleanup
echo -e "\n${YELLOW}Current Seed-OSS-36B resources:${NC}"
echo "Deployments:"
oc get deployment -l app=seed-oss-36b 2>/dev/null || echo "  None found"
echo "Pods:"
oc get pods -l app=seed-oss-36b 2>/dev/null || echo "  None found"
echo "Services:"
oc get service seed-oss-36b-service 2>/dev/null || echo "  None found"
echo "Routes:"
oc get route seed-oss-36b-route 2>/dev/null || echo "  None found"
echo "ConfigMaps:"
oc get configmap seed-oss-36b-config 2>/dev/null || echo "  None found"
echo "Secrets:"
oc get secret seed-oss-api-key seed-oss-hf-token 2>/dev/null || echo "  None found"

# Confirmation prompt
echo ""
read -p "$(echo -e ${RED}Are you sure you want to delete all Seed-OSS-36B resources? [y/N]: ${NC})" -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cleanup cancelled by user${NC}"
    exit 0
fi

echo -e "\n${YELLOW}Starting cleanup...${NC}"

# Delete resources using the all-in-one deployment file if it exists
if [ -f "../deployments/seed-oss-36b-deployment.yaml" ]; then
    echo -e "${YELLOW}Deleting resources using seed-oss-36b-deployment.yaml...${NC}"
    if oc delete -f ../deployments/seed-oss-36b-deployment.yaml --ignore-not-found=true; then
        echo -e "${GREEN}✓${NC} Resources deleted via manifest"
    else
        echo -e "${YELLOW}⚠${NC} Some resources may not have been deleted via manifest, trying individual deletion..."
    fi
fi

# Individual resource cleanup (backup method)
echo -e "\n${YELLOW}Ensuring all resources are removed...${NC}"

# Delete deployment
if oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
    echo -e "${YELLOW}Deleting deployment...${NC}"
    oc delete deployment seed-oss-36b-vllm --ignore-not-found=true
    echo -e "${GREEN}✓${NC} Deployment deleted"
fi

# Delete service
if oc get service seed-oss-36b-service >/dev/null 2>&1; then
    echo -e "${YELLOW}Deleting service...${NC}"
    oc delete service seed-oss-36b-service --ignore-not-found=true
    echo -e "${GREEN}✓${NC} Service deleted"
fi

# Delete route
if oc get route seed-oss-36b-route >/dev/null 2>&1; then
    echo -e "${YELLOW}Deleting route...${NC}"
    oc delete route seed-oss-36b-route --ignore-not-found=true
    echo -e "${GREEN}✓${NC} Route deleted"
fi

# Delete configmap
if oc get configmap seed-oss-36b-config >/dev/null 2>&1; then
    echo -e "${YELLOW}Deleting configmap...${NC}"
    oc delete configmap seed-oss-36b-config --ignore-not-found=true
    echo -e "${GREEN}✓${NC} ConfigMap deleted"
fi

# Ask about secrets
echo ""
read -p "$(echo -e ${YELLOW}Do you want to delete the API key and HuggingFace token secrets? [y/N]: ${NC})" -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Delete API key secret
    if oc get secret seed-oss-api-key >/dev/null 2>&1; then
        echo -e "${YELLOW}Deleting API key secret...${NC}"
        oc delete secret seed-oss-api-key --ignore-not-found=true
        echo -e "${GREEN}✓${NC} API key secret deleted"
    fi
    
    # Delete HuggingFace token secret
    if oc get secret seed-oss-hf-token >/dev/null 2>&1; then
        echo -e "${YELLOW}Deleting HuggingFace token secret...${NC}"
        oc delete secret seed-oss-hf-token --ignore-not-found=true
        echo -e "${GREEN}✓${NC} HuggingFace token secret deleted"
    fi
else
    echo -e "${BLUE}Secrets preserved${NC}"
fi

# Wait for pods to terminate
echo -e "\n${YELLOW}Waiting for pods to terminate...${NC}"
oc wait --for=delete pods -l app=seed-oss-36b --timeout=120s 2>/dev/null || echo -e "${YELLOW}⚠${NC} Timeout waiting for pods to terminate (this is normal)"

# Final verification
echo -e "\n${YELLOW}Verifying cleanup...${NC}"
REMAINING_RESOURCES=0

if oc get deployment seed-oss-36b-vllm >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} Deployment still exists"
    REMAINING_RESOURCES=1
fi

if oc get pods -l app=seed-oss-36b >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠${NC} Some pods may still be terminating"
fi

if oc get service seed-oss-36b-service >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} Service still exists"
    REMAINING_RESOURCES=1
fi

if oc get route seed-oss-36b-route >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} Route still exists"
    REMAINING_RESOURCES=1
fi

if oc get configmap seed-oss-36b-config >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} ConfigMap still exists"
    REMAINING_RESOURCES=1
fi

# Summary
if [ $REMAINING_RESOURCES -eq 0 ]; then
    echo -e "\n${GREEN}=== Cleanup Complete ===${NC}"
    echo -e "${GREEN}✓${NC} All Seed-OSS-36B resources have been removed"
    echo -e "${GREEN}✓${NC} API endpoint is no longer accessible"
    echo -e "${GREEN}✓${NC} GPU resources are now available for other workloads"
else
    echo -e "\n${YELLOW}=== Cleanup Partially Complete ===${NC}"
    echo -e "${YELLOW}⚠${NC} Some resources may still exist. Check manually:"
    echo "oc get all,configmap,secret -l app=seed-oss-36b -n $NAMESPACE"
fi

echo ""
echo -e "${BLUE}To redeploy, run: ./deploy.sh deploy${NC}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "Check remaining resources: oc get all -l app=seed-oss-36b -n $NAMESPACE"
echo "View namespace resources: oc get all -n $NAMESPACE"
echo "Force delete stuck pods: oc delete pods -l app=seed-oss-36b --force --grace-period=0 -n $NAMESPACE"