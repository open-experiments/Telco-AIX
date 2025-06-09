# Telecom AI Agent Framework

A cloud-native framework for autonomous network management through cooperating AI agents that detect, diagnose, plan, and resolve telecom network issues without human intervention. 

## What Problem Does This Solve?

Traditional network management requires human operators to:
1. Monitor for issues
2. Diagnose root causes
3. Plan remediation
4. Execute changes
5. Verify solutions

The Telecom AI Agent Framework automates this entire workflow through specialized AI agents that work together to maintain network health autonomously, reducing downtime and operational costs.

<div align="center">
    <img src="https://raw.githubusercontent.com/open-experiments/Telco-AIX/refs/heads/main/agentic/images/arc.png" width="400"/>
</div>

## Key Features

- **Autonomous Network Management**: End-to-end automation from detection to resolution
- **Specialized Agent Types**: Four agent types that perform distinct roles:
  - **Diagnostic Agents**: Monitor telemetry and detect anomalies
  - **Planning Agents**: Generate remediation plans for detected issues
  - **Execution Agents**: Apply configuration changes to network elements
  - **Validation Agents**: Verify that issues have been resolved
- **Communication Infrastructure**:
  - **Model Context Protocol (MCP)**: Standardized context sharing between agents
  - **Agent Communication Protocol (ACP)**: Direct agent-to-agent messaging
- **Orchestration Layer**: Coordinates workflow execution and agent activities
- **Real-time Dashboard**: Visualize agent activities, network telemetry, and workflow status
- **Cloud-Native Design**: Built following 12-factor app methodology

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/open-experiments/Telco-AIX.git
cd Telco-AIX/agentic

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install WebSocket support (required for agent communication)
pip install websockets
```

### Running the Framework

To start the framework with the full agent ecosystem and dashboard:

```bash
python main.py --run-test
```

This will:
1. Start the MCP server, ACP broker, and dashboard server
2. Create and initialize all four agent types
3. Register agents with the orchestration service
4. Create a sample workflow
5. Run a test scenario that demonstrates the complete anomaly detection and resolution process

Then open your browser and navigate to:
```
http://localhost:8080
```

**✅ System Status: Fully Functional** - All WebSocket connection issues have been resolved and the system works seamlessly.

![](https://raw.githubusercontent.com/open-experiments/Telco-AIX/refs/heads/main/agentic/images/agentic.png)<br>

You'll see the dashboard with real-time visualization of agent activities, network telemetry, and workflow status.

### Command Line Options

- `--mcp-host`: Host address for the MCP server (default: 0.0.0.0)
- `--mcp-port`: Port for the MCP server (default: 8000)
- `--acp-host`: Host address for the ACP broker (default: 0.0.0.0)
- `--acp-port`: Port for the ACP broker (default: 8002)
- `--dashboard-host`: Host address for the dashboard (default: 0.0.0.0)
- `--dashboard-port`: Port for the dashboard (default: 8080)
- `--run-test`: Execute a test network anomaly scenario

## Dashboard Only Mode

If you want to see just the dashboard visualization without the agent communications:

```bash
python dashboard_mode.py
```

This runs a simplified version with simulated data that demonstrates what the system looks like when operational.

## Architecture

The framework is built around these core components:

1. **Agent Core Framework**
   - Base agent abstractions with lifecycle management
   - Protocol implementations (MCP, ACP)
   - Context assembly and management

2. **Specialized Agents**
   - Network Diagnostic Agent: Monitors telemetry for anomalies
   - Planning Agent: Generates solutions for identified issues
   - Execution Agent: Implements planned solutions
   - Validation Agent: Verifies solution effectiveness

3. **Communication Protocols**
   - Model Context Protocol (MCP): Context sharing with headers, metadata, and payloads
   - Agent Communication Protocol (ACP): Direct agent-to-agent messaging

4. **Orchestration Service**
   - Agent discovery and registration
   - Workflow management
   - Task decomposition
   - Resource allocation

5. **Dashboard Server**
   - Real-time visualization of network telemetry
   - Agent status monitoring
   - Workflow progress tracking
   - Event timeline display

## How It Works: Network Anomaly Resolution

This framework implements autonomous network issue resolution through the following workflow:

1. **Detect**: Diagnostic Agent continuously monitors telemetry data and detects network anomalies (e.g., high packet loss)
2. **Share Context**: Anomaly details are stored in MCP with relevant metrics and telemetry
3. **Notify**: Diagnostic Agent notifies Planning Agents via ACP
4. **Plan**: Planning Agent generates a resolution plan based on the context
5. **Execute**: Execution Agent implements the solution (e.g., routing updates, config changes)
6. **Validate**: Validation Agent verifies the issue is resolved

The entire workflow is managed by the orchestration service, which ensures proper sequencing and handles errors or timeouts.

## Troubleshooting

### Prerequisites
Make sure you have installed the WebSocket support:
```bash
pip install websockets
```

### Service URLs
- **MCP Server**: http://localhost:8000
- **ACP Broker**: http://localhost:8002  
- **Dashboard**: http://localhost:8080

### Health Checks
You can verify services are running:
```bash
# Check ACP broker health
curl http://localhost:8002/health

# Check if agents are connected
curl http://localhost:8002/agents
```

### Recent Fixes Applied
- ✅ **WebSocket Connection Issues**: All services now use `127.0.0.1` instead of `localhost` to avoid DNS resolution conflicts
- ✅ **Service Startup**: Added health check verification before agent initialization
- ✅ **Timeout Issues**: Enhanced retry logic and connection timeouts
- ✅ **Python Compatibility**: Updated deprecated `datetime.utcnow()` calls for future Python versions
- ✅ **Dashboard Issues**: Fixed agent details and workflow details pages with proper API endpoints
- ✅ **Data Serialization**: Resolved JSON serialization issues with datetime and numpy data types

## Extending the Framework

### Creating a New Agent Type

1. Create a new class extending the appropriate base agent type:

```python
from agent.base import PlanningAgent

class CustomPlanningAgent(PlanningAgent):
    async def _initialize(self) -> None:
        # Custom initialization logic
        pass
        
    async def _run(self) -> None:
        # Custom processing logic
        pass
        
    async def _shutdown(self) -> None:
        # Custom shutdown logic
        pass
        
    async def generate_plan(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        # Planning logic
        return {"plan": "..."}
```

2. Register the agent with the orchestration service

### Creating a New Workflow Type

Extend the orchestration service with a new workflow implementation:

```python
async def _execute_custom_workflow(self, workflow: Dict[str, Any]) -> None:
    # Implement workflow steps
    pass
```

## Project Structure

```
agentic/
├── agent/                 # Base agent classes
├── agents/                # Specialized agent implementations
│   ├── diagnostic/        # Diagnostic agents
│   ├── planning/          # Planning agents
│   ├── execution/         # Execution agents
│   └── validation/        # Validation agents
├── protocols/             # Communication protocols
│   ├── mcp/               # Model Context Protocol
│   └── acp/               # Agent Communication Protocol
├── orchestration/         # Orchestration services
├── dashboard/             # Dashboard web interface
├── main.py                # Main application
├── dashboard_mode.py      # Dashboard-only mode
├── CLAUDE.md              # AI assistant guidance for development
└── requirements.txt       # Python dependencies
```

## For AI Assistants

This project includes a `CLAUDE.md` file that provides specific guidance for AI assistants working with this codebase, including:
- Architecture overview and key components
- Essential commands and troubleshooting steps  
- Development workflow information
- Implementation notes and best practices

## For AI Assistants

This project includes a `CLAUDE.md` file that provides specific guidance for AI assistants working with this codebase, including:
- Architecture overview and key components
- Essential commands and troubleshooting steps  
- Development workflow information
- Implementation notes and best practices

'''
CLAUDE.md file is not available in repo
'''

## Disclaimer

In the current (1st drop) implementation we have leveraged rule based agentic approach rather than FM enclosed inside the agent (as that would be heavy to launch in our sandbox due to humble lab we have). In coming version(s) it will evolve more towards self-learning and improving beings with help of a custom small fm engine (wip).
