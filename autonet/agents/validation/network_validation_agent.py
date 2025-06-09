# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from agent.base import ValidationAgent
from protocols.mcp.client import MCPClient
from protocols.mcp.schema import ContextType, ConfidenceLevel
from protocols.acp.messaging import ACPMessagingClient
from protocols.acp.schema import MessageType, ActionType, MessagePriority

class NetworkValidationAgent(ValidationAgent):
    """Agent for validating network changes and resolution of issues."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None,
                 mcp_url: str = "http://localhost:8000",
                 acp_broker_url: str = "http://localhost:8002",
                 telemetry_url: str = "http://localhost:8001/telemetry",
                 config: Dict[str, Any] = None):
        """Initialize a new network validation agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            mcp_url: URL for the MCP server
            acp_broker_url: URL for the ACP message broker
            telemetry_url: URL for the telemetry API
            config: Additional configuration
        """
        super().__init__(
            agent_id=agent_id,
            name=name or "Network Validation Agent",
            description="Agent for validating network changes and resolution of issues",
            config=config or {}
        )
        
        self.mcp_url = mcp_url
        self.acp_broker_url = acp_broker_url
        self.telemetry_url = telemetry_url
        
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
            ActionType.VALIDATE,
            self._handle_validate_action
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
            self.logger.info(f"Validation agent task cancelled")
        
    async def _shutdown(self) -> None:
        """Shut down the agent."""
        if self.mcp_client:
            await self.mcp_client.close()
            
        if self.acp_client:
            await self.acp_client.close()
            
        self.logger.info(f"Shut down {self.name}")
        
    async def _handle_notification(self, message) -> None:
        """Handle an incoming notification message.
        
        Args:
            message: The notification message
        """
        if message.action_type == ActionType.VALIDATE:
            # Extract validation information
            plan_id = message.payload.get("plan_id", "")
            status = message.payload.get("status", "")
            context_refs = message.context_refs
            
            if context_refs:
                # Retrieve execution result from MCP
                execution_result = await self._retrieve_context(context_refs[0])
                
                # Retrieve original plan and anomaly
                plan_context = await self._retrieve_related_context(plan_id, "plan")
                anomaly_context = await self._retrieve_related_context(plan_id, "anomaly")
                
                if execution_result and plan_context:
                    # Validate the execution
                    validation_task = asyncio.create_task(
                        self.validate_execution(execution_result, plan_context, anomaly_context)
                    )
                    
                    # We don't await the task here to allow it to run asynchronously
                    self.logger.info(f"Started validation of plan {plan_id} execution")
        
    async def _handle_request(self, message) -> None:
        """Handle an incoming request message.
        
        Args:
            message: The request message
        """
        if message.action_type == ActionType.VALIDATE:
            # Handle validation request
            payload = message.payload
            
            # Validate execution based on request
            validation_result = await self.validate_execution(
                payload.get("execution_result", {}),
                payload.get("original_plan", {}),
                payload.get("anomaly_context", {})
            )
            
            # Send response
            response = self.acp_client.create_message(
                recipient_id=message.header.sender_id,
                message_type=MessageType.RESPONSE,
                action_type=ActionType.VALIDATE,
                payload=validation_result,
                priority=message.header.priority,
                in_reply_to=message.header.message_id,
                conversation_id=message.header.conversation_id,
                context_refs=message.context_refs
            )
            
            await self.acp_client.send_message(response)
    
    async def _handle_validate_action(self, message) -> None:
        """Handle a message with a VALIDATE action type.
        
        Args:
            message: The message with VALIDATE action
        """
        # Generic handler for validate actions
        self.logger.info(f"Received validate action in message {message.header.message_id}")
        
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
            
    async def _retrieve_related_context(self, correlation_id: str, context_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve a related context by correlation ID and type.
        
        Args:
            correlation_id: Correlation ID to search for
            context_type: Type of context to retrieve (e.g., "plan", "anomaly")
            
        Returns:
            The context data, or None if not found
        """
        try:
            # Query for contexts with the correlation ID
            query = {
                "correlation_id": correlation_id,
                "intent_tags": [context_type] if context_type != "plan" else ["action_plan", "network_resolution"]
            }
            
            response = await self.mcp_client.query_context(query)
            
            if response and "contexts" in response:
                contexts = response["contexts"]
                if contexts:
                    # Return the first matching context
                    first_id = next(iter(contexts))
                    return contexts[first_id].get("payload", {})
            
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving {context_type} context for {correlation_id}: {str(e)}")
            return None
        
    async def _store_validation_result(self, result: Dict[str, Any], correlation_id: str) -> str:
        """Store validation result in MCP.
        
        Args:
            result: The validation result
            correlation_id: Correlation ID for related contexts
            
        Returns:
            ID of the stored context
        """
        # Create context
        context = self.mcp_client.create_context(
            domain_type=ContextType.PROCEDURAL,
            payload={
                "validation_result": result,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=3,
            confidence=ConfidenceLevel.HIGH,
            correlation_id=correlation_id,
            intent_tags=["validation_result", "network_operations"],
            scope=result.get("scope")
        )
        
        # Store in MCP
        response = await self.mcp_client.send_context(context)
        context_id = response.get("context_id", "")
        self.logger.info(f"Stored validation result in MCP with ID {context_id}")
        
        return context_id
        
    async def _notify_completion(self, result: Dict[str, Any]) -> None:
        """Notify orchestrator about validation completion.
        
        Args:
            result: The validation result
        """
        # For simplicity, we'll broadcast to orchestrators
        message = self.acp_client.create_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            action_type=ActionType.INFORM,
            payload={
                "operation": "validation_completed",
                "plan_id": result.get("plan_id", ""),
                "validation_status": result.get("validation_status", "unknown"),
                "issue_resolved": result.get("issue_resolved", False),
                "validation_time": result.get("validation_time"),
                "scope": result.get("scope", "")
            },
            priority=MessagePriority.MEDIUM,
            context_refs=[result.get("context_id", "")]
        )
        
        await self.acp_client.send_message(message)
        self.logger.info(f"Notified about validation completion for plan {result.get('plan_id', '')}")
        
    async def _fetch_current_telemetry(self, scope: str) -> Dict[str, Any]:
        """Fetch current telemetry data for validation.
        
        Args:
            scope: Scope of telemetry to fetch (e.g., cell site ID)
            
        Returns:
            Current telemetry data
        """
        # In a real implementation, this would make an API call to get telemetry data
        # For this example, we'll simulate telemetry data
        
        # Generate baseline values (representing improved metrics after remediation)
        telemetry = {
            "packet_loss": 0.005,     # 0.5% packet loss (improved)
            "latency": 30.0,          # 30ms latency (improved)
            "throughput": 1100.0,     # 1100 Mbps (improved)
            "jitter": 2.0,            # 2ms jitter (improved)
            "signal_strength": -60.0  # -60 dBm (improved)
        }
        
        # Add timestamp and metadata
        telemetry["timestamp"] = datetime.utcnow().isoformat()
        telemetry["cell_id"] = scope
        telemetry["region"] = scope.split("_")[0] if "_" in scope else "UNKNOWN"
        
        self.logger.debug(f"Fetched current telemetry data: {telemetry}")
        return telemetry
        
    async def validate_execution(self, 
                               execution_result: Dict[str, Any], 
                               original_plan: Dict[str, Any],
                               anomaly_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate the execution of a plan.
        
        Args:
            execution_result: The result of executing a plan
            original_plan: The original plan that was executed
            anomaly_context: Original anomaly context that triggered the plan
            
        Returns:
            Validation results
        """
        plan_id = original_plan.get("id", execution_result.get("plan_id", "unknown"))
        scope = execution_result.get("scope", "")
        
        self.logger.info(f"Validating execution of plan {plan_id}")
        
        # Initialize validation result
        result = {
            "plan_id": plan_id,
            "validation_status": "in_progress",
            "validation_time": datetime.utcnow().isoformat(),
            "executed_by": execution_result.get("executed_by", "unknown"),
            "validated_by": self.agent_id,
            "scope": scope,
            "metrics_validation": [],
            "step_validations": []
        }
        
        # Check execution status
        execution_status = execution_result.get("status", "unknown")
        if execution_status not in ["completed", "partial"]:
            result.update({
                "validation_status": "failed",
                "issue_resolved": False,
                "reason": f"Execution was not completed successfully (status: {execution_status})"
            })
            
            # Store result in MCP
            context_id = await self._store_validation_result(result, plan_id)
            result["context_id"] = context_id
            
            # Notify completion
            await self._notify_completion(result)
            
            return result
        
        # Get original anomalies if available
        anomalies = []
        if anomaly_context:
            anomalies = anomaly_context.get("anomalies", [])
        
        # Get current telemetry data
        current_telemetry = await self._fetch_current_telemetry(scope)
        
        # Validate metrics
        metrics_validated = True
        for anomaly in anomalies:
            metric_name = anomaly.get("metric", "")
            original_value = anomaly.get("value", 0.0)
            
            if metric_name and metric_name in current_telemetry:
                current_value = current_telemetry[metric_name]
                improvement = self._calculate_improvement(metric_name, original_value, current_value)
                is_resolved = self._is_metric_resolved(metric_name, current_value)
                
                metric_validation = {
                    "metric": metric_name,
                    "original_value": original_value,
                    "current_value": current_value,
                    "improvement_percentage": improvement,
                    "is_resolved": is_resolved
                }
                
                result["metrics_validation"].append(metric_validation)
                
                if not is_resolved:
                    metrics_validated = False
            else:
                # Can't validate this metric
                result["metrics_validation"].append({
                    "metric": metric_name,
                    "error": "Metric not found in current telemetry",
                    "is_resolved": False
                })
                metrics_validated = False
        
        # Validate plan steps
        steps_executed = execution_result.get("steps_total", 0)
        steps_completed = execution_result.get("steps_completed", 0)
        steps_failed = execution_result.get("steps_failed", 0)
        
        step_results = execution_result.get("step_results", [])
        for step_result in step_results:
            step_id = step_result.get("step_id", "")
            status = step_result.get("status", "")
            
            # Find original step definition
            original_step = None
            for step in original_plan.get("steps", []):
                if step.get("id") == step_id:
                    original_step = step
                    break
            
            step_validation = {
                "step_id": step_id,
                "name": step_result.get("name", ""),
                "status": status,
                "is_valid": status == "success" or status == "skipped",
                "duration": step_result.get("duration", 0)
            }
            
            result["step_validations"].append(step_validation)
        
        # Determine final validation status
        steps_valid = (steps_completed + steps_failed == steps_executed)
        issue_resolved = metrics_validated and steps_valid
        
        result.update({
            "validation_status": "completed",
            "issue_resolved": issue_resolved,
            "steps_valid": steps_valid,
            "metrics_valid": metrics_validated,
            "completion_percentage": (steps_completed / steps_executed * 100) if steps_executed > 0 else 0
        })
        
        # Store result in MCP
        context_id = await self._store_validation_result(result, plan_id)
        result["context_id"] = context_id
        
        # Notify completion
        await self._notify_completion(result)
        
        return result
        
    def _calculate_improvement(self, metric: str, original: float, current: float) -> float:
        """Calculate the improvement percentage for a metric.
        
        Args:
            metric: Name of the metric
            original: Original value
            current: Current value
            
        Returns:
            Improvement percentage
        """
        if original == 0:
            return 0.0
            
        # Different metrics improve in different directions
        if metric in ["packet_loss", "latency", "jitter"]:
            # Lower is better
            if original <= current:
                return 0.0
            return ((original - current) / original) * 100
        else:
            # Higher is better
            if original >= current:
                return 0.0
            return ((current - original) / original) * 100
    
    def _is_metric_resolved(self, metric: str, value: float) -> bool:
        """Determine if a metric is within acceptable ranges.
        
        Args:
            metric: Name of the metric
            value: Current value
            
        Returns:
            True if the metric is within acceptable ranges
        """
        # Define acceptable thresholds
        thresholds = {
            "packet_loss": 0.01,     # 1% packet loss or less is acceptable
            "latency": 50.0,         # 50ms or less is acceptable
            "throughput": 900.0,     # 900 Mbps or more is acceptable
            "jitter": 5.0,           # 5ms or less is acceptable
            "signal_strength": -70.0  # -70 dBm or better is acceptable
        }
        
        if metric not in thresholds:
            return True  # Can't validate unknown metrics
            
        threshold = thresholds[metric]
        
        # Check based on metric type
        if metric in ["packet_loss", "latency", "jitter"]:
            # Lower is better
            return value <= threshold
        else:
            # Higher is better
            return value >= threshold
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        # Handle different message types
        if message.get("message_type") == "request" and message.get("action_type") == "validate":
            return await self.validate_execution(
                message.get("execution_result", {}),
                message.get("original_plan", {}),
                message.get("anomaly_context", {})
            )
        
        # Default response for other message types
        return {
            "status": "acknowledged",
            "timestamp": datetime.utcnow().isoformat(),
            "validation_agent": self.agent_id
        }
        
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities supported by this agent."""
        return [
            "execution_validation",
            "telemetry_verification",
            "network_health_assessment",
            "resolution_confirmation"
        ]
