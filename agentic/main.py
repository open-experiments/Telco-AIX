# agentic/main.py

import asyncio
import logging
import argparse
import sys
from datetime import datetime, timezone
import os

from protocols.mcp.server import MCPServer
from protocols.acp.broker import ACPMessageBroker
from orchestration.service import OrchestrationService
from dashboard.server import DashboardServer
from agents.diagnostic.network_diagnostic_agent import NetworkDiagnosticAgent
from agents.planning.network_planning_agent import NetworkPlanningAgent
from agents.execution.network_execution_agent import NetworkExecutionAgent
from agents.validation.network_validation_agent import NetworkValidationAgent
from protocols.acp.schema import AgentDescription, CapabilityInfo, ActionType, MessageType, MessagePriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("agentic")

def check_websocket_support():
    """Check if WebSocket libraries are installed and available."""
    try:
        import websockets
        logger.info("Found websockets library")
        return True
    except ImportError:
        try:
            import wsproto
            logger.info("Found wsproto library")
            return True
        except ImportError:
            logger.warning("No WebSocket library detected! Communication between agents will fail.")
            logger.warning("Please install WebSocket support: pip install uvicorn[standard]")
            return False

async def setup_services(mcp_host="0.0.0.0", mcp_port=8000, 
                       acp_host="0.0.0.0", acp_port=8002,
                       dashboard_host="0.0.0.0", dashboard_port=8080):
    """Set up the core services for the telecom agent framework."""
    
    logger.info("Initializing core services...")
    
    # Create orchestration service
    orchestration = OrchestrationService()
    await orchestration.initialize()
    logger.info("Orchestration service initialized")
    
    # Start MCP server
    mcp_server = MCPServer(host=mcp_host, port=mcp_port)
    mcp_task = asyncio.create_task(mcp_server.start())
    logger.info(f"MCP server starting on {mcp_host}:{mcp_port}")
    
    # Start ACP broker
    acp_broker = ACPMessageBroker(host=acp_host, port=acp_port)
    acp_task = asyncio.create_task(acp_broker.start())
    logger.info(f"ACP broker starting on {acp_host}:{acp_port}")
    
    # Wait for servers to initialize and verify they're ready
    logger.info("Waiting for services to initialize...")
    await asyncio.sleep(3)  # Initial wait
    
    # Check if services are ready with retries
    import aiohttp
    max_retries = 10
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                # Test the ACP broker health endpoint
                async with session.get(f"http://{acp_host}:{acp_port}/health", timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status == 200:
                        logger.info("ACP broker is ready")
                        # Give the WebSocket server a moment to fully initialize
                        await asyncio.sleep(2)
                        break
                    else:
                        logger.warning(f"ACP broker health check returned status {response.status}")
        except Exception as e:
            logger.warning(f"ACP broker not ready yet (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error("ACP broker failed to become ready")
    
    # Start dashboard server
    dashboard_server = DashboardServer(
        host=dashboard_host,
        port=dashboard_port,
        mcp_url=f"http://127.0.0.1:{mcp_port}",
        acp_url=f"http://127.0.0.1:{acp_port}",
        orchestration_service=orchestration
    )
    dashboard_task = asyncio.create_task(dashboard_server.start())
    logger.info(f"Dashboard server starting on {dashboard_host}:{dashboard_port}")
    
    # Wait for dashboard to initialize
    await asyncio.sleep(2)
    
    return mcp_server, acp_broker, orchestration, dashboard_server, mcp_task, acp_task, dashboard_task

async def setup_agents(mcp_url="http://127.0.0.1:8000", acp_broker_url="ws://127.0.0.1:8002"):
    """Set up all agent types."""
    agents = []
    
    logger.info("Setting up agents...")
    
    try:
        # Create and initialize diagnostic agent
        logger.info("Setting up Diagnostic Agent...")
        try:
            diagnostic_agent = NetworkDiagnosticAgent(
                name="Primary Network Diagnostic Agent",
                telemetry_url="http://127.0.0.1:8001/telemetry",
                mcp_url=mcp_url,
                acp_broker_url=acp_broker_url,
                poll_interval=10.0  # For demonstration, poll every 10 seconds
            )
            await diagnostic_agent.initialize()
            await diagnostic_agent.start()
            agents.append(diagnostic_agent)
            logger.info("Diagnostic Agent initialized and started")
        except Exception as e:
            logger.error(f"Failed to set up Diagnostic Agent: {str(e)}")
        
        # Create and initialize planning agent
        logger.info("Setting up Planning Agent...")
        try:
            planning_agent = NetworkPlanningAgent(
                name="Primary Network Planning Agent",
                mcp_url=mcp_url,
                acp_broker_url=acp_broker_url
            )
            await planning_agent.initialize()
            await planning_agent.start()
            agents.append(planning_agent)
            logger.info("Planning Agent initialized and started")
        except Exception as e:
            logger.error(f"Failed to set up Planning Agent: {str(e)}")
        
        # Create and initialize execution agent
        logger.info("Setting up Execution Agent...")
        try:
            execution_agent = NetworkExecutionAgent(
                name="Primary Network Execution Agent",
                mcp_url=mcp_url,
                acp_broker_url=acp_broker_url,
                network_api_url="http://127.0.0.1:8003/api/network"
            )
            await execution_agent.initialize()
            await execution_agent.start()
            agents.append(execution_agent)
            logger.info("Execution Agent initialized and started")
        except Exception as e:
            logger.error(f"Failed to set up Execution Agent: {str(e)}")
        
        # Create and initialize validation agent
        logger.info("Setting up Validation Agent...")
        try:
            validation_agent = NetworkValidationAgent(
                name="Primary Network Validation Agent",
                mcp_url=mcp_url,
                acp_broker_url=acp_broker_url,
                telemetry_url="http://127.0.0.1:8001/telemetry"
            )
            await validation_agent.initialize()
            await validation_agent.start()
            agents.append(validation_agent)
            logger.info("Validation Agent initialized and started")
        except Exception as e:
            logger.error(f"Failed to set up Validation Agent: {str(e)}")
        
        logger.info(f"Successfully set up {len(agents)} out of 4 agents")
        
    except Exception as e:
        logger.error(f"Error in agent setup: {str(e)}")
    
    return agents

async def register_with_orchestration(agents, orchestration):
    """Register all agents with the orchestration service."""
    
    if not agents:
        logger.warning("No agents to register with orchestration")
        return
    
    # Define capabilities for each agent type
    capabilities = {
        "diagnostic": CapabilityInfo(
            action_types=[ActionType.DIAGNOSE],
            domains=["network", "anomaly_detection"],
            description="Network diagnostic agent for detecting anomalies in telemetry data"
        ),
        "planning": CapabilityInfo(
            action_types=[ActionType.PLAN],
            domains=["network", "anomaly_resolution"],
            description="Network planning agent for generating resolution plans"
        ),
        "execution": CapabilityInfo(
            action_types=[ActionType.EXECUTE],
            domains=["network", "configuration", "automation"],
            description="Network execution agent for implementing resolution plans"
        ),
        "validation": CapabilityInfo(
            action_types=[ActionType.VALIDATE],
            domains=["network", "verification"],
            description="Network validation agent for verifying resolutions"
        )
    }
    
    # Register each agent
    for agent in agents:
        try:
            agent_type = agent.agent_type.value
            
            # Create agent description for registration
            description = AgentDescription(
                agent_id=agent.agent_id,
                agent_type=agent_type,
                name=agent.name,
                capabilities=capabilities.get(agent_type, CapabilityInfo(
                    action_types=[],
                    domains=[],
                    description="Unknown agent type"
                )),
                network_location={
                    "host": "localhost",
                    "port": 8080
                },
                status="active",
                last_seen=datetime.now(timezone.utc)
            )
            
            # Register with orchestration service
            await orchestration.register_agent(description)
            logger.info(f"Registered {agent.name} ({agent.agent_id}) with orchestration")
        except Exception as e:
            logger.error(f"Failed to register agent {agent.name}: {str(e)}")

async def ensure_dashboard_directories():
    """Ensure the dashboard directories for templates and static files exist."""
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create dashboard directories if they don't exist
        dashboard_dir = os.path.join(base_dir, "dashboard")
        os.makedirs(os.path.join(dashboard_dir, "templates"), exist_ok=True)
        os.makedirs(os.path.join(dashboard_dir, "static", "css"), exist_ok=True)
        os.makedirs(os.path.join(dashboard_dir, "static", "js"), exist_ok=True)
        
        logger.info("Dashboard directories created/verified")
    except Exception as e:
        logger.error(f"Error ensuring dashboard directories: {str(e)}")

async def run_network_anomaly_test(agents, dashboard_server):
    """Run a test of the network anomaly resolution workflow."""
    if not agents:
        logger.error("Cannot run test scenario: No agents available")
        return
    
    try:
        diagnostic_agent = agents[0]  # First agent is the diagnostic agent
        
        # Step 1: Simulate an anomaly detection
        logger.info("=== TEST SCENARIO: Network Anomaly Detection and Resolution ===")
        
        # Create anomaly data
        anomaly = {
            "metric": "packet_loss",
            "value": 0.05,  # 5% packet loss
            "z_score": 4.2,
            "description": "Anomalous high packet loss detected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cell_id": "SITE_123",
            "region": "REGION_A"
        }
        
        telemetry = {
            "packet_loss": 0.05,
            "latency": 65.0,
            "throughput": 850.0,
            "jitter": 8.0,
            "signal_strength": -75.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cell_id": "SITE_123",
            "region": "REGION_A"
        }
        
        logger.info("Sending initial telemetry to dashboard...")
        # Send telemetry update to dashboard
        await dashboard_server.broadcast_update("telemetry_update", telemetry)
        
        # Add an event
        await dashboard_server.broadcast_update("event", {
            "agent_type": "diagnostic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Anomaly detected: High packet loss (5%) on cell site SITE_123"
        })
        
        # Step 2: Process the anomaly
        logger.info(f"Diagnostic Agent detected anomaly: {anomaly['description']}")
        
        # Simulate anomaly detection processing
        anomalies = [anomaly]
        try:
            result = await diagnostic_agent._report_anomalies(anomalies, telemetry)
            logger.info("Anomaly reported successfully")
        except Exception as e:
            logger.error(f"Error reporting anomaly: {str(e)}")
        
        await asyncio.sleep(2)  # Wait for notification to be processed
        
        # Add events
        await dashboard_server.broadcast_update("event", {
            "agent_type": "planning",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Generated plan to resolve packet loss issue on cell site SITE_123"
        })
        
        await asyncio.sleep(1)
        
        await dashboard_server.broadcast_update("event", {
            "agent_type": "execution",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Executing network adjustments to reduce packet loss"
        })
        
        await asyncio.sleep(2)
        
        # Update telemetry to show improvement
        improved_telemetry = {
            "packet_loss": 0.005,  # Improved to 0.5%
            "latency": 35.0,
            "throughput": 1050.0,
            "jitter": 3.0,
            "signal_strength": -65.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cell_id": "SITE_123",
            "region": "REGION_A"
        }
        
        await dashboard_server.broadcast_update("telemetry_update", improved_telemetry)
        
        await asyncio.sleep(1)
        
        await dashboard_server.broadcast_update("event", {
            "agent_type": "validation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Validated solution: Packet loss reduced from 5% to 0.5%"
        })
        
        logger.info("=== TEST SCENARIO COMPLETED ===")
        logger.info("Dashboard available at: http://localhost:8080")
    except Exception as e:
        logger.error(f"Error running test scenario: {str(e)}")

async def main():
    """Main entry point for the telecom agent framework."""
    
    parser = argparse.ArgumentParser(description="Telecom Agent Framework")
    parser.add_argument("--mcp-host", default="0.0.0.0", help="MCP server host")
    parser.add_argument("--mcp-port", type=int, default=8000, help="MCP server port")
    parser.add_argument("--acp-host", default="0.0.0.0", help="ACP broker host")
    parser.add_argument("--acp-port", type=int, default=8002, help="ACP broker port")
    parser.add_argument("--dashboard-host", default="0.0.0.0", help="Dashboard server host")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard server port")
    parser.add_argument("--run-test", action="store_true", help="Run the sample test scenario")
    args = parser.parse_args()
    
    logger.info("Starting Telecom Agent Framework")
    
    # Check for WebSocket support
    has_websocket = check_websocket_support()
    if not has_websocket:
        print("WebSocket libraries missing! Please run: pip install uvicorn[standard]")
        print("Continuing anyway, but agent communication will likely fail...")
    
    # Ensure dashboard directories exist
    await ensure_dashboard_directories()
    
    # Initialize variables to avoid errors in the finally block
    agents = []
    orchestration = None
    dashboard_server = None
    
    try:
        # Set up services
        mcp_server, acp_broker, orchestration, dashboard_server, mcp_task, acp_task, dashboard_task = await setup_services(
            mcp_host=args.mcp_host,
            mcp_port=args.mcp_port,
            acp_host=args.acp_host,
            acp_port=args.acp_port,
            dashboard_host=args.dashboard_host,
            dashboard_port=args.dashboard_port
        )
        
        # Set up agents
        agents = await setup_agents(
            mcp_url=f"http://127.0.0.1:{args.mcp_port}",
            acp_broker_url=f"ws://127.0.0.1:{args.acp_port}"
        )
        
        if agents:
            # Register agents with orchestration
            await register_with_orchestration(agents, orchestration)
            
            # Create a sample workflow via orchestration
            workflow_id = await orchestration.create_workflow(
                workflow_type="anomaly_resolution",
                parameters={
                    "priority": "high",
                    "source": "automated"
                },
                initiator_id=agents[0].agent_id  # Use diagnostic agent as initiator
            )
            
            logger.info(f"Created workflow {workflow_id}")
            
            # Run test scenario if requested
            if args.run_test and dashboard_server is not None:
                test_task = asyncio.create_task(run_network_anomaly_test(agents, dashboard_server))
        else:
            logger.warning("No agents were set up successfully. Skipping workflow creation and testing.")
        
        # Wait for services to run
        try:
            logger.info("All components initialized. Press Ctrl+C to exit.")
            # Wait for MCP, ACP, and dashboard tasks
            await asyncio.gather(mcp_task, acp_task, dashboard_task)
        except asyncio.CancelledError:
            logger.info("Services cancelled")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
    finally:
        # Shutdown services
        if orchestration:
            try:
                await orchestration.shutdown()
                logger.info("Orchestration service shut down")
            except Exception as e:
                logger.error(f"Error shutting down orchestration: {str(e)}")
        
        # Stop all agents
        for agent in agents:
            try:
                await agent.stop()
                logger.info(f"Agent {agent.name} stopped")
            except Exception as e:
                logger.error(f"Error stopping agent {agent.name}: {str(e)}")
        
        logger.info("Telecom Agent Framework stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")