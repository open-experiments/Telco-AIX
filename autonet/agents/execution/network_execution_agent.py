# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from agent.base import ExecutionAgent
from protocols.mcp.client import MCPClient
from protocols.mcp.schema import ContextType, ConfidenceLevel
from protocols.acp.messaging import ACPMessagingClient
from protocols.acp.schema import MessageType, ActionType, MessagePriority

class NetworkExecutionAgent(ExecutionAgent):
    """Agent for executing network configuration and optimization plans."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None,
                 mcp_url: str = "http://localhost:8000",
                 acp_broker_url: str = "http://localhost:8002",
                 network_api_url: str = "http://localhost:8003/api/network",
                 config: Dict[str, Any] = None):
        """Initialize a new network execution agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            mcp_url: URL for the MCP server
            acp_broker_url: URL for the ACP message broker
            network_api_url: URL for the network API
            config: Additional configuration
        """
        super().__init__(
            agent_id=agent_id,
            name=name or "Network Execution Agent",
            description="Agent for executing network configuration and optimization plans",
            config=config or {}
        )
        
        self.mcp_url = mcp_url
        self.acp_broker_url = acp_broker_url
        self.network_api_url = network_api_url
        
        # Clients for communication
        self.mcp_client = None
        self.acp_client = None
        
        # Current execution status
        self.executing_plans = {}
        
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
            ActionType.EXECUTE,
            self._handle_execute_action
        )
        
        self.logger.info(f"Initialized {self.name}")
        
    async def _run(self) -> None:
        """Run the agent's main processing loop."""
        self.logger.info(f"Starting {self.name}")
        
        # Wait indefinitely (agent is reactive to messages)
        try:
            while True:
                await asyncio.sleep(60)  # Check on executing plans every minute
                await self._check_executing_plans()
        except asyncio.CancelledError:
            self.logger.info(f"Execution agent task cancelled")
        
    async def _shutdown(self) -> None:
        """Shut down the agent."""
        if self.mcp_client:
            await self.mcp_client.close()
            
        if self.acp_client:
            await self.acp_client.close()
            
        self.logger.info(f"Shut down {self.name}")
        
    async def _check_executing_plans(self) -> None:
        """Check on the status of executing plans and update as needed."""
        plans_to_remove = []
        
        for plan_id, plan_data in self.executing_plans.items():
            # Check if the plan has timed out
            current_time = datetime.utcnow()
            start_time = datetime.fromisoformat(plan_data["start_time"])
            timeout = plan_data.get("timeout", 3600)  # Default 1 hour timeout
            
            if (current_time - start_time).total_seconds() > timeout:
                # Plan has timed out
                self.logger.warning(f"Plan {plan_id} execution timed out after {timeout} seconds")
                
                # Update plan status
                plan_data["status"] = "failed"
                plan_data["error"] = f"Execution timed out after {timeout} seconds"
                
                # Store result in MCP
                await self._store_execution_result(plan_id, plan_data)
                
                # Notify validation agents
                await self._notify_validation_agents(plan_id, plan_data)
                
                # Mark for removal
                plans_to_remove.append(plan_id)
                
        # Remove completed or timed out plans
        for plan_id in plans_to_remove:
            del self.executing_plans[plan_id]
        
    async def _handle_notification(self, message) -> None:
        """Handle an incoming notification message.
        
        Args:
            message: The notification message
        """
        if message.action_type == ActionType.EXECUTE:
            # Extract plan information
            plan_summary = message.payload.get("plan_summary", "")
            priority = message.payload.get("priority", "medium")
            context_refs = message.context_refs
            
            if context_refs:
                # Retrieve full plan from MCP
                plan_data = await self._retrieve_context(context_refs[0])
                
                if plan_data:
                    # Execute the plan
                    execution_task = asyncio.create_task(
                        self.execute_plan(plan_data)
                    )
                    
                    # We don't await the task here to allow it to run asynchronously
                    self.logger.info(f"Started execution of plan {plan_data.get('id', 'unknown')} with priority {priority}")
        
    async def _handle_request(self, message) -> None:
        """Handle an incoming request message.
        
        Args:
            message: The request message
        """
        if message.action_type == ActionType.EXECUTE:
            # Handle execution request
            payload = message.payload
            
            # Execute plan based on request
            result = await self.execute_plan(payload)
            
            # Send response
            response = self.acp_client.create_message(
                recipient_id=message.header.sender_id,
                message_type=MessageType.RESPONSE,
                action_type=ActionType.EXECUTE,
                payload=result,
                priority=message.header.priority,
                in_reply_to=message.header.message_id,
                conversation_id=message.header.conversation_id,
                context_refs=message.context_refs
            )
            
            await self.acp_client.send_message(response)
    
    async def _handle_execute_action(self, message) -> None:
        """Handle a message with an EXECUTE action type.
        
        Args:
            message: The message with EXECUTE action
        """
        # Generic handler for execute actions
        self.logger.info(f"Received execute action in message {message.header.message_id}")
        
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
        
    async def _store_execution_result(self, plan_id: str, result: Dict[str, Any]) -> str:
        """Store execution result in MCP.
        
        Args:
            plan_id: ID of the executed plan
            result: The execution result
            
        Returns:
            ID of the stored context
        """
        # Create context
        context = self.mcp_client.create_context(
            domain_type=ContextType.PROCEDURAL,
            payload={
                "execution_result": result,
                "plan_id": plan_id,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=3,
            confidence=ConfidenceLevel.HIGH,
            correlation_id=plan_id,
            intent_tags=["execution_result", "network_operations"],
            scope=result.get("scope")
        )
        
        # Store in MCP
        response = await self.mcp_client.send_context(context)
        context_id = response.get("context_id", "")
        self.logger.info(f"Stored execution result in MCP with ID {context_id}")
        
        return context_id
        
    async def _notify_validation_agents(self, plan_id: str, result: Dict[str, Any]) -> None:
        """Notify validation agents about completed execution.
        
        Args:
            plan_id: ID of the executed plan
            result: The execution result
        """
        # For simplicity, we'll broadcast to all validation agents
        message = self.acp_client.create_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            action_type=ActionType.VALIDATE,
            payload={
                "plan_id": plan_id,
                "status": result.get("status", "unknown"),
                "execution_time": result.get("end_time"),
                "scope": result.get("scope", "")
            },
            priority=MessagePriority.HIGH,
            context_refs=[result.get("context_id", "")]
        )
        
        await self.acp_client.send_message(message)
        self.logger.info(f"Notified validation agents about completed execution of plan {plan_id}")
        
    async def execute_network_command(self, command: str, target: str) -> Dict[str, Any]:
        """Execute a network command.
        
        Args:
            command: The command to execute
            target: The target device or system
            
        Returns:
            The command result
        """
        # In a real implementation, this would interact with network devices
        # For this example, we'll simulate successful command execution
        
        self.logger.info(f"Executing command '{command}' on target '{target}'")
        
        # Simulate command execution delay
        await asyncio.sleep(1.0)
        
        # Return simulated result
        return {
            "status": "success",
            "command": command,
            "target": target,
            "timestamp": datetime.utcnow().isoformat(),
            "output": f"Simulated output for {command} on {target}"
        }
        
    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the provided plan.
        
        Args:
            plan: The plan to execute
            
        Returns:
            The result of the execution
        """
        plan_id = plan.get("id", str(uuid.uuid4()))
        steps = plan.get("steps", [])
        
        if not steps:
            return {
                "plan_id": plan_id,
                "status": "failed",
                "error": "Plan contains no steps to execute",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "executed_by": self.agent_id
            }
            
        # Initialize execution result
        result = {
            "plan_id": plan_id,
            "status": "in_progress",
            "start_time": datetime.utcnow().isoformat(),
            "executed_by": self.agent_id,
            "steps_total": len(steps),
            "steps_completed": 0,
            "steps_failed": 0,
            "step_results": [],
            "scope": plan.get("scope", "")
        }
        
        # Store in executing plans
        self.executing_plans[plan_id] = result
        
        # Execute each step
        for step in steps:
            step_id = step.get("id", str(uuid.uuid4()))
            step_name = step.get("name", "Unnamed step")
            action_type = step.get("action_type", "unknown")
            target = step.get("target", "")
            command = step.get("command", "")
            
            self.logger.info(f"Executing step {step_name} ({step_id}) of plan {plan_id}")
            
            step_result = {
                "step_id": step_id,
                "name": step_name,
                "status": "in_progress",
                "start_time": datetime.utcnow().isoformat()
            }
            
            try:
                # Execute the step based on action type
                if action_type == "diagnostic" or action_type == "configuration":
                    # Execute network command
                    command_result = await self.execute_network_command(command, target)
                    step_result.update({
                        "status": command_result.get("status", "unknown"),
                        "output": command_result.get("output", ""),
                        "command": command,
                        "target": target
                    })
                elif action_type == "validation":
                    # Skip validation steps (these are for the validation agent)
                    step_result.update({
                        "status": "skipped",
                        "reason": "Validation steps are executed by validation agents"
                    })
                else:
                    # Unknown action type
                    step_result.update({
                        "status": "skipped",
                        "reason": f"Unknown action type: {action_type}"
                    })
                
                # Mark step as completed
                if step_result.get("status") == "success":
                    result["steps_completed"] += 1
                elif step_result.get("status") == "failed":
                    result["steps_failed"] += 1
                
            except Exception as e:
                # Handle step execution error
                step_result.update({
                    "status": "failed",
                    "error": str(e)
                })
                result["steps_failed"] += 1
            finally:
                # Finalize step result
                step_result["end_time"] = datetime.utcnow().isoformat()
                step_result["duration"] = (datetime.fromisoformat(step_result["end_time"]) - 
                                         datetime.fromisoformat(step_result["start_time"])).total_seconds()
                result["step_results"].append(step_result)
                
                # Update overall status
                if result["steps_failed"] > 0:
                    result["status"] = "partial"
                
                # Update executing plans
                self.executing_plans[plan_id] = result
        
        # Finalize execution result
        end_time = datetime.utcnow().isoformat()
        result.update({
            "status": "completed" if result["steps_failed"] == 0 else "partial",
            "end_time": end_time,
            "duration": (datetime.fromisoformat(end_time) - 
                        datetime.fromisoformat(result["start_time"])).total_seconds()
        })
        
        # Store result in MCP
        context_id = await self._store_execution_result(plan_id, result)
        result["context_id"] = context_id
        
        # Notify validation agents
        await self._notify_validation_agents(plan_id, result)
        
        # Remove from executing plans
        del self.executing_plans[plan_id]
        
        return result
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        # Handle different message types
        if message.get("message_type") == "request" and message.get("action_type") == "execute":
            return await self.execute_plan(message.get("payload", {}))
        
        # Default response for other message types
        return {
            "status": "acknowledged",
            "timestamp": datetime.utcnow().isoformat(),
            "execution_agent": self.agent_id
        }
        
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities supported by this agent."""
        return [
            "network_configuration",
            "command_execution",
            "plan_execution",
            "network_automation"
        ]
