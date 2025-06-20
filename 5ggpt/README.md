# 5G Components management with LLM

# 5G-GPT: AI-Driven Kubernetes-Based Management of Open5GS

Open5GS-LLM is a Kubernetes-native application that leverages Large Language Models (LLMs) to monitor, manage, and explain Open5GS deployments. It provides a conversational interface—similar to ChatGPT—that enables users to query Open5GS status, update configurations, and gain insight into 4G/5G core functions, all through natural language.

## Project Goals

- Deploy and manage Open5GS on Kubernetes clusters
- Use LLMs to interpret and edit Kubernetes manifests and Helm values
- Explain Open5GS network functions (AMF, SMF, UPF, etc.) and configuration options
- Provide a ChatGPT-like web UI for interactive management and learning
- Enable dynamic configuration updates using natural language commands

## Architecture

Web Chat UI 
<--> LLM Backend (OpenAI, Aws Bedrock, Local LLM)  
<--> Kubernetes Cluster (Open5GS )  
      ↑  
      |  
Kubernetes Controller  
 - Watches Open5GS Pods, CRDs, and Helm changes  
 - Converts LLM instructions to safe Kubernetes actions
Open5Gs Controller 
 - Watches the logs for Open5G components 
 - Configure and update Open5Gs components  


## Features

- Real-time Monitoring  
  - AMF, SMF, UPF, HSS status and logs  
  - UE session tracking via Kubernetes events

- Natural Language Configuration  
  - Convert natural language to YAML or Helm changes  
  - Apply changes to running clusters with safety checks

- Conversational Explanations  
  - Ask about config values and their impact  
  - LLM provides context-aware descriptions

- Ccross layer corelation use cases  
  - Auto scaling 
  - Auto healing
  - Service assurace like a NOC KPI reporting 

## Components

TODO:

## Getting Started

### Prerequisites

TODO:
