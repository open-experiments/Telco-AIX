# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Telco-AIX is a collection of independent AI/ML experiment projects for the telecommunications domain. Each subdirectory is a self-contained project with its own dependencies — there is no monorepo build system, shared library, or CI/CD pipeline.

Models and datasets are published to HuggingFace: `huggingface.co/collections/fenar/telco-aix-66737384ab5687fe3d9a4b94`

## Running Projects

Each project is independent. The general pattern is:

```bash
cd <project-dir>
pip install -r requirements.txt
python main.py  # or the project's entry script
```

**agentic/** and **autonet/** (multi-agent frameworks):
```bash
pip install -r requirements.txt
python main.py          # starts MCP (port 8000), ACP broker (8002), dashboard (8080)
python main.py --run-test  # agentic: runs with test scenario
```

**telco-sme/** (Gradio UI):
```bash
pip install -r requirements-v2.txt
python sme-web-ui-v2.py
```

**5gprod/** (Dash dashboard), **crm/** (Flask), **churn/** (Flask server on port 5000), **revenueassurance/** (Flask): each has a main Python script that starts a web server.

Projects with Dockerfiles: `crm/`, `churn/`, `etc/model-as-a-server/`

## Testing

Only `agentic/` and `autonet/` have pytest tests:
```bash
cd agentic && pytest
cd autonet && pytest
```

No global test runner or linting configuration exists.

## Architecture

### Protocol-Based Agent Frameworks (agentic/, autonet/)

These are the most complex projects, implementing a multi-agent architecture with two custom protocols:

- **MCP (Model Context Protocol)**: HTTP/FastAPI-based context and data sharing between agents
- **ACP (Agent Communication Protocol)**: WebSocket-based inter-agent messaging with a central broker

Four agent types follow a workflow pipeline: **Diagnostic** -> **Planning** -> **Execution** -> **Validation**. Each inherits from a base `Agent` class with a state machine (INITIALIZING -> IDLE -> PROCESSING -> WAITING -> TERMINATED/ERROR). The `orchestration/service.py` coordinates agent workflows.

`autonet/` extends `agentic/` with real NOC integration, Ansible playbook execution, and multi-backend LLM support (Anthropic, OpenAI, HuggingFace, local) via `protocols/mcp/backends/`.

### Common Tech Stack Across Projects

- **Web UIs**: Flask, Gradio, Dash, FastAPI (varies by project)
- **ML/DL**: PyTorch, TensorFlow/Keras, scikit-learn, HuggingFace Transformers
- **LLM/RAG**: LangChain, OpenAI API, FAISS vector search
- **Model Export**: ONNX runtime for inference optimization
- **Data**: pandas, numpy; visualization with matplotlib/seaborn/plotly
- **Async**: asyncio, websockets, aiohttp (in agent frameworks)
- **Validation**: Pydantic (in agent frameworks)
- **Config**: python-dotenv for environment variables

### Project Domains

| Project | What It Does |
|---------|-------------|
| `agentic/`, `autonet/` | Multi-agent telco network automation (MCP/ACP protocols) |
| `5gnetops/` | 5G fault prediction with BERT/MoE models |
| `5gprod/` | NOC dashboard with LLM-augmented operations |
| `telco-sme/` | Telco knowledge portal with embeddings search |
| `airan-energy/` | AI-RAN energy optimization (DQN + traffic forecasting) |
| `intclass/` | Intent classification with fine-tuned Qwen models |
| `llm-rca/` | Root cause analysis with LLM chaining and RAG |
| `churn/` | Customer churn prediction pipeline |
| `revenueassurance/` | Revenue assurance with Random Forest and Transformer models |
| `secops/`, `iot-sec/` | Network and IoT security ML models |
