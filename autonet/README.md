#  Autonomous 5G Network 

## 🚀 Overview

An Autonomous 5G Networks featuring real-time anomaly detection, autonomous remediation, and advanced agent topology visualization. This system monitors AMF, SMF, and UPF components, automatically detects anomalies, and executes remediation playbooks without human intervention.

**Demo Video**: [Watch on YouTube](https://www.youtube.com/watch?v=nQlEBPeQ1hk)

## ✨ Key Features

### Autonomous Operations
- **Real-time Anomaly Detection**: Continuously monitors 5G network components (AMF, SMF, UPF)
- **Intelligent Workflow Management**: Multi-agent system with diagnostic, planning, execution, and validation stages
- **Automated Remediation**: Executes Ansible playbooks based on anomaly patterns
- **Timeline-based Processing**: Replay historical data with adjustable speed (1x, 5x, 10x, 100x)

### Advanced Visualization
- **Agent Topology View**: Real-time visualization of agent interactions and performance
- **Performance Metrics**: Live charts for registration rates, session establishment, throughput, and latency
- **Workflow Tracking**: Detailed view of active and historical remediation workflows
- **Predictive Analytics**: Bottleneck predictions and health scoring for each agent

### Technical Capabilities
- **Scalable Architecture**: Modular design with async Python backend
- **RESTful API**: Comprehensive API for integration with external systems
- **Real-time WebSocket Updates**: Live dashboard updates without polling
- **Configurable Playbooks**: Extensible remediation library

## 🏗️ Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   NOC Data Monitor  │────▶│  Workflow Manager   │────▶│ Ansible Executor    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Metrics Tracker    │     │  Agent Topology     │
└─────────────────────┘     │     Manager         │
           │                └─────────────────────┘
           ▼                           │
                                       ▼
┌──────────────────────────────────────────────────┐
│              Web Dashboard (Async HTTP)          │
└──────────────────────────────────────────────────┘
```

### Agent Workflow
1. **Diagnostic Agent**: Analyzes anomaly patterns and severity
2. **Planning Agent**: Generates remediation strategies
3. **Execution Agent**: Runs selected Ansible playbooks
4. **Validation Agent**: Verifies successful remediation

## 📋 Prerequisites

- RHOCP 4.18+
- RHOAI Operator 2.20+
- Python 3.8+
- Ansible 
- Modern web browser (Chrome, Firefox, Safari)

## 🔧 Installation

**Clone the repository**
```bash
git clone https://github.com/yourusername/Telco-AIX.git
cd Telco-AIX/agentic
```

 **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage
```bash
cd Telco-AIX/autonet
python main.py --data-path processed_data --playbook-dir playbooks
```

## 🌐 Accessing the Dashboard

Once running, access the dashboard at:
```
http://localhost:30080
```
If you are using with RHOCPAI , you will need a OCP Service and Route:
```
kind: Route
apiVersion: route.openshift.io/v1
metadata:
  name: agentic-dashboard
  namespace: tme-aix
spec:
  path: /
  to:
    kind: Service
    name: agentic-dashboard
    weight: 100
  port:
    targetPort: 30080
---
kind: Service
apiVersion: v1
metadata:
  name: agentic-dashboard
  namespace: tme-aix
spec:
  ports:
    - protocol: TCP
      port: 30080
      targetPort: 30080
  selector:
    statefulset: tme-aix-wb01 <-- This is your RHOAI WorkBench Name (POD)

```

### Dashboard Views
1. **System Overview**: Real-time statistics and anomaly counts

![New NOC Dashboard](https://raw.githubusercontent.com/open-experiments/Telco-AIX/refs/heads/main/autonet/images/n1.png)

3. **Agent Topology**: Interactive visualization of agent network
   
![New NOC Dashboard](https://raw.githubusercontent.com/open-experiments/Telco-AIX/refs/heads/main/autonet/images/n2.png)

5. **Active Workflows**: Current and historical remediation workflows

![New NOC Dashboard](https://raw.githubusercontent.com/open-experiments/Telco-AIX/refs/heads/main/autonet/images/n3.png)


### Playbook Structure
Place Ansible playbooks in the `playbooks` directory:
```yaml
# Example: scale_amf_resources.yml
---
- name: Scale AMF Resources
  hosts: amf_nodes
  tasks:
    - name: Increase CPU allocation
      # Your remediation tasks here
```

### Provided Playbooks
- `scale_amf_resources.yml` - Scale AMF CPU/Memory
- `restart_amf_service.yml` - Restart AMF services
- `restart_smf_service.yml` - Restart SMF services  
- `adjust_upf_load_balancing.yml` - Adjust UPF load balancing
- `resource_optimization.yml` - General optimization

