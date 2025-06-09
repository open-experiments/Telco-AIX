# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from agent.base import PlanningAgent
from protocols.mcp.client import MCPClient
from protocols.mcp.schema import ContextType, ConfidenceLevel
from protocols.acp.messaging import ACPMessagingClient
from protocols.acp.schema import MessageType, ActionType, MessagePriority

class NetworkPlanningAgent(PlanningAgent):
    """Agent for generating plans to resolve network issues."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None,
                 mcp_url: str = "http://127.0.0.1:8000",
                 acp_broker_url: str = "ws://127.0.0.1:8002",
                 config: Dict[str, Any] = None):
        """Initialize a new network planning agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            mcp_url: URL for the MCP server
            acp_broker_url: URL for the ACP message broker
            config: Additional configuration
        """
        super().__init__(
            agent_id=agent_id,
            name=name or "Network Planning Agent",
            description="Agent for generating plans to resolve network issues",
            config=config or {}
        )
        
        self.mcp_url = mcp_url
        self.acp_broker_url = acp_broker_url
        
        # Clients for communication
        self.mcp_client = None
        self.acp_client = None
        
    async def _initialize(self) -> None:
        """Initialize the agent."""
        # Initialize MCP client
        self.mcp_client = MCPClient(
            client_id=self.agent_id,
            base_url=self.mcp_url
        )
        await self.mcp_client.initialize()
        
        # Initialize ACP client
        self.acp_client = ACPMessagingClient(
            agent_id=self.agent_id,
            broker_url=self.acp_broker_url
        )
        await self.acp_client.initialize()
        
        # Register ACP message handlers
        self.acp_client.register_message_handler(
            MessageType.NOTIFICATION,
            self._handle_notification
        )
        
        self.acp_client.register_message_handler(
            MessageType.REQUEST,
            self._handle_request
        )
        
        self.acp_client.register_action_handler(
            ActionType.PLAN,
            self._handle_plan_action
        )
        
        self.logger.info(f"Initialized {self.name}")
        
    async def _run(self) -> None:
        """Run the agent's main processing loop."""
        self.logger.info(f"Starting {self.name}")
        
        # Wait indefinitely (agent is reactive to messages)
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour
        except asyncio.CancelledError:
            self.logger.info(f"Planning agent task cancelled")
        
    async def _shutdown(self) -> None:
        """Shut down the agent."""
        if self.mcp_client:
            await self.mcp_client.close()
            
        if self.acp_client:
            await self.acp_client.close()
            
        self.logger.info(f"Shut down {self.name}")
        
    async def _handle_notification(self, message: Dict[str, Any]) -> None:
        """Handle an incoming notification message.
        
        Args:
            message: The notification message
        """
        if message.action_type == ActionType.PLAN:
            # Extract anomaly information
            anomalies = message.payload.get("anomalies", [])
            context_refs = message.context_refs
            
            if anomalies and context_refs:
                # Retrieve full context from MCP
                context_data = await self._retrieve_context(context_refs[0])
                
                if context_data:
                    # Generate a plan
                    plan = await self.generate_plan({
                        "anomalies": anomalies,
                        "context": context_data,
                        "severity": message.payload.get("severity", "medium"),
                        "source_agent": message.header.sender_id
                    })
                    
                    # Store plan in MCP
                    plan_context = self.mcp_client.create_context(
                        domain_type=ContextType.PROCEDURAL,
                        payload=plan,
                        priority=3,
                        confidence=ConfidenceLevel.HIGH,
                        correlation_id=context_refs[0],
                        intent_tags=["network_resolution", "action_plan"],
                        scope=context_data.get("telemetry", {}).get("cell_id")
                    )
                    
                    response = await self.mcp_client.send_context(plan_context)
                    plan_context_id = response.get("context_id")
                    
                    # Notify execution agents
                    await self._notify_execution_agents(plan, plan_context_id)
        
    async def _handle_request(self, message: Dict[str, Any]) -> None:
        """Handle an incoming request message.
        
        Args:
            message: The request message
        """
        if message.action_type == ActionType.PLAN:
            # Handle planning request
            payload = message.payload
            
            # Generate plan based on request
            plan = await self.generate_plan(payload)
            
            # Send response
            response = self.acp_client.create_message(
                recipient_id=message.header.sender_id,
                message_type=MessageType.RESPONSE,
                action_type=ActionType.PLAN,
                payload=plan,
                priority=message.header.priority,
                in_reply_to=message.header.message_id,
                conversation_id=message.header.conversation_id,
                context_refs=message.context_refs
            )
            
            await self.acp_client.send_message(response)
    
    async def _handle_plan_action(self, message: Dict[str, Any]) -> None:
        """Handle a message with a PLAN action type.
        
        Args:
            message: The message with PLAN action
        """
        # Generic handler for plan actions
        self.logger.info(f"Received plan action in message {message.header.message_id}")
        
    async def _retrieve_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve context from MCP.
        
        Args:
            context_id: ID of the context to retrieve
            
        Returns:
            The context data, or None if not found
        """
        try:
            response = await self.mcp_client.query_context({"context_id": context_id})
            if response and "contexts" in response:
                contexts = response["contexts"]
                if context_id in contexts:
                    return contexts[context_id].get("payload", {})
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving context {context_id}: {str(e)}")
            return None
        
    async def _notify_execution_agents(self, plan: Dict[str, Any], plan_context_id: str) -> None:
        """Notify execution agents about a new plan.
        
        Args:
            plan: The generated plan
            plan_context_id: ID of the plan context in MCP
        """
        # For simplicity, we'll broadcast to all execution agents
        message = self.acp_client.create_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            action_type=ActionType.EXECUTE,
            payload={
                "plan_summary": plan.get("summary", "Network issue resolution plan"),
                "priority": plan.get("priority", "medium"),
                "generation_time": datetime.now(timezone.utc).isoformat()
            },
            priority=MessagePriority.HIGH,
            context_refs=[plan_context_id]
        )
        
        await self.acp_client.send_message(message)
        self.logger.info(f"Notified execution agents about new plan")
        
    async def generate_plan(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan to address the described problem.
        
        Args:
            problem_context: The context describing the problem to solve
            
        Returns:
            A plan with steps to address the problem
        """
        # Extract information from the problem context
        anomalies = problem_context.get("anomalies", [])
        severity = problem_context.get("severity", "medium")
        
        if not anomalies:
            return {
                "error": "No anomalies provided to generate a plan",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "planning_agent": self.agent_id
            }
        
        # Generate a plan based on the anomalies
        plan = {
            "id": str(uuid.uuid4()),
            "summary": f"Plan to resolve {len(anomalies)} network anomalies",
            "severity": severity,
            "priority": self._calculate_priority(anomalies, severity),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": self.agent_id,
            "steps": [],
            "estimated_duration": 0,  # Will be calculated based on steps
            "resources_required": []
        }
        
        # Add steps based on anomaly types
        total_duration = 0
        for anomaly in anomalies:
            metric = anomaly.get("metric")
            steps = self._generate_steps_for_anomaly(metric, anomaly)
            
            for step in steps:
                plan["steps"].append(step)
                total_duration += step.get("estimated_duration", 0)
                
                # Add required resources
                for resource in step.get("resources", []):
                    if resource not in plan["resources_required"]:
                        plan["resources_required"].append(resource)
        
        plan["estimated_duration"] = total_duration
        
        return plan
    
    def _calculate_priority(self, anomalies: List[Dict[str, Any]], severity: str) -> str:
        """Calculate the priority of the plan.
        
        Args:
            anomalies: The anomalies to address
            severity: The reported severity
            
        Returns:
            Priority level (high, medium, or low)
        """
        # In a real implementation, this would use more sophisticated logic
        # For this example, we'll use a simple heuristic
        if severity == "high" or any(a.get("z_score", 0) > 5.0 for a in anomalies):
            return "high"
        elif severity == "medium" or any(a.get("z_score", 0) > 3.0 for a in anomalies):
            return "medium"
        else:
            return "low"
    
    def _generate_steps_for_anomaly(self, metric: str, anomaly: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate steps to address a specific anomaly.
        
        Args:
            metric: The metric with the anomaly
            anomaly: The anomaly details
            
        Returns:
            List of steps to address the anomaly
        """
        steps = []
        
        # Generate steps based on the metric type
        if metric == "packet_loss":
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Check network interface errors",
                "description": "Examine interface counters for errors and drops",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "show interface counters errors",
                "estimated_duration": 60,  # seconds
                "resources": ["network_cli"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Verify routing configuration",
                "description": "Check for routing loops or misconfigurations",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "show ip route",
                "estimated_duration": 60,  # seconds
                "resources": ["network_cli"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Adjust queue management",
                "description": "Optimize QoS settings to reduce packet drops",
                "action_type": "configuration",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "configure qos optimize anti-congestion",
                "estimated_duration": 120,  # seconds
                "resources": ["network_cli", "configuration_management"]
            })
            
        elif metric == "latency":
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Check for congestion",
                "description": "Analyze traffic patterns for congestion points",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "show traffic analysis",
                "estimated_duration": 90,  # seconds
                "resources": ["traffic_analyzer"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Optimize routing",
                "description": "Adjust routing to minimize latency",
                "action_type": "configuration",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "optimize-route --target=latency",
                "estimated_duration": 180,  # seconds
                "resources": ["network_cli", "route_optimizer"]
            })
            
        elif metric == "throughput":
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Check link utilization",
                "description": "Analyze bandwidth utilization across links",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "show interface utilization",
                "estimated_duration": 60,  # seconds
                "resources": ["network_cli"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Identify top talkers",
                "description": "Find applications or hosts consuming most bandwidth",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "show top-talkers",
                "estimated_duration": 90,  # seconds
                "resources": ["traffic_analyzer"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Adjust traffic shaping",
                "description": "Optimize bandwidth allocation",
                "action_type": "configuration",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "configure traffic-shape optimize",
                "estimated_duration": 150,  # seconds
                "resources": ["network_cli", "configuration_management"]
            })
            
        elif metric == "jitter":
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Analyze traffic patterns",
                "description": "Identify sources of variable delay",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "analyze-jitter --detail",
                "estimated_duration": 120,  # seconds
                "resources": ["traffic_analyzer"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Optimize QoS for real-time traffic",
                "description": "Adjust QoS to prioritize consistent delay for real-time traffic",
                "action_type": "configuration",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "configure qos prioritize real-time",
                "estimated_duration": 180,  # seconds
                "resources": ["network_cli", "configuration_management"]
            })
            
        elif metric == "signal_strength":
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Check antenna alignment",
                "description": "Verify antenna alignment and signals",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "check-antenna-alignment",
                "estimated_duration": 300,  # seconds
                "resources": ["rf_analyzer"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": "Adjust power levels",
                "description": "Optimize transmit power settings",
                "action_type": "configuration",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": "adjust-power-levels --optimize",
                "estimated_duration": 240,  # seconds
                "resources": ["network_cli", "rf_management"]
            })
            
        else:
            # Generic steps for unknown metrics
            steps.append({
                "id": str(uuid.uuid4()),
                "name": f"Analyze {metric} anomaly",
                "description": f"Gather more information about the {metric} anomaly",
                "action_type": "diagnostic",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": f"analyze-anomaly --metric={metric}",
                "estimated_duration": 120,  # seconds
                "resources": ["network_analyzer"]
            })
            
            steps.append({
                "id": str(uuid.uuid4()),
                "name": f"Apply standard remediation for {metric}",
                "description": f"Apply standard fixes for {metric} issues",
                "action_type": "configuration",
                "target": f"site:{anomaly.get('cell_id')}",
                "command": f"auto-remediate --metric={metric}",
                "estimated_duration": 180,  # seconds
                "resources": ["network_cli", "auto_remediation"]
            })
        
        # Add a verification step for all anomalies
        steps.append({
            "id": str(uuid.uuid4()),
            "name": f"Verify {metric} improvement",
            "description": f"Check that {metric} values have returned to normal range",
            "action_type": "validation",
            "target": f"site:{anomaly.get('cell_id')}",
            "command": f"validate-metric --name={metric}",
            "estimated_duration": 120,  # seconds
            "resources": ["telemetry_analyzer"]
        })
        
        return steps
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        # Handle different message types
        if message.get("message_type") == "request" and message.get("action_type") == "plan":
            return await self.generate_plan(message.get("payload", {}))
        
        # Default response for other message types
        return {
            "status": "acknowledged",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "planning_agent": self.agent_id
        }
        
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities supported by this agent."""
        return [
            "network_planning",
            "anomaly_resolution_planning",
            "resource_optimization",
            "configuration_generation"
        ]
