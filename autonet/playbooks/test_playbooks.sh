#!/bin/bash
echo "🧪 Testing Ansible Playbooks..."

# Test each playbook with demo variables
echo "Testing AMF scaling playbook..."
ansible-playbook scale_amf_resources.yml -e "anomaly_id=test_001 cpu_limit=3000m memory_limit=6Gi severity=HIGH"

echo -e "\nTesting SMF restart playbook..."
ansible-playbook restart_smf_service.yml -e "anomaly_id=test_002 severity=MEDIUM"

echo -e "\nTesting UPF load balancing playbook..."
ansible-playbook adjust_upf_load_balancing.yml -e "anomaly_id=test_003 threshold=75 severity=HIGH"

echo -e "\nTesting resource optimization playbook..."
ansible-playbook resource_optimization.yml -e "anomaly_id=test_004 severity=CRITICAL"

echo -e "\n✅ All playbook tests completed!"
