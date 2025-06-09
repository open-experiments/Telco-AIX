# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from agent.base import DiagnosticAgent
from protocols.mcp.client import MCPClient
from protocols.mcp.schema import ContextType, ConfidenceLevel
from protocols.acp.messaging import ACPMessagingClient
from protocols.acp.schema import MessageType, ActionType, MessagePriority

class NetworkMetric:
    """Class representing a network metric with historical data for anomaly detection."""
    
    def __init__(self, name: str, window_size: int = 100, threshold_z: float = 3.0):
        """Initialize a network metric.
        
        Args:
            name: Name of the metric
            window_size: Size of the historical window to keep
            threshold_z: Z-score threshold for anomaly detection
        """
        self.name = name
        self.window_size = window_size
        self.threshold_z = threshold_z
        self.values = []
        self.timestamps = []
        
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new value to the metric.
        
        Args:
            value: The metric value
            timestamp: Timestamp for the value (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        # Keep only window_size most recent values
        if len(self.values) > self.window_size:
            self.values.pop(0)
            self.timestamps.pop(0)
            
    def detect_anomaly(self) -> Tuple[bool, float, Optional[str]]:
        """Detect if the current value is anomalous.
        
        Returns:
            Tuple of (is_anomaly, z_score, description)
        """
        if len(self.values) < 10:  # Need some history for meaningful detection
            return False, 0.0, None
            
        current = self.values[-1]
        history = self.values[:-1]
        
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:  # Avoid division by zero
            return False, 0.0, None
            
        z_score = abs((current - mean) / std)
        
        if z_score > self.threshold_z:
            direction = "high" if current > mean else "low"
            description = f"Anomalous {direction} value detected for {self.name}: {current} (z-score: {z_score:.2f})"
            return True, z_score, description
        else:
            return False, z_score, None
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical information about this metric.
        
        Returns:
            Dictionary of statistical information
        """
        if not self.values:
            return {
                "name": self.name,
                "count": 0
            }
            
        return {
            "name": self.name,
            "count": len(self.values),
            "current": self.values[-1],
            "mean": np.mean(self.values),
            "std": np.std(self.values),
            "min": np.min(self.values),
            "max": np.max(self.values),
            "timestamp": self.timestamps[-1].isoformat()
        }

class NetworkDiagnosticAgent(DiagnosticAgent):
    """Agent for diagnosing network issues based on telemetry data."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None, 
                 telemetry_url: str = "http://localhost:8001/telemetry",
                 mcp_url: str = "http://localhost:8000",
                 acp_broker_url: str = "http://localhost:8002",
                 metrics_config: Optional[Dict[str, Dict[str, Any]]] = None,
                 poll_interval: float = 60.0,
                 config: Dict[str, Any] = None):
        """Initialize a new network diagnostic agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            telemetry_url: URL for the telemetry API
            mcp_url: URL for the MCP server
            acp_broker_url: URL for the ACP message broker
            metrics_config: Configuration for metrics to monitor
            poll_interval: Interval between polling for new telemetry data (seconds)
            config: Additional configuration
        """
        super().__init__(
            agent_id=agent_id, 
            name=name or "Network Diagnostic Agent",
            description="Agent for diagnosing network issues based on telemetry data",
            config=config or {}
        )
        
        self.telemetry_url = telemetry_url
        self.mcp_url = mcp_url
        self.acp_broker_url = acp_broker_url
        self.poll_interval = poll_interval
        
        # Set up metrics to monitor
        self.metrics: Dict[str, NetworkMetric] = {}
        metrics_config = metrics_config or {
            "packet_loss": {"threshold_z": 3.0},
            "latency": {"threshold_z": 3.0},
            "throughput": {"threshold_z": 3.0},
            "jitter": {"threshold_z": 3.0},
            "signal_strength": {"threshold_z": 3.0}
        }
        
        for metric_name, config in metrics_config.items():
            self.metrics[metric_name] = NetworkMetric(
                name=metric_name,
                window_size=config.get("window_size", 100),
                threshold_z=config.get("threshold_z", 3.0)
            )
            
        # Clients for communication
        self.mcp_client = None
        self.acp_client = None
        
        # Polling task
        self.polling_task = None
        
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
            MessageType.REQUEST,
            self._handle_request
        )
        
        self.logger.info(f"Initialized {self.name} with {len(self.metrics)} metrics to monitor")
        
    async def _run(self) -> None:
        """Run the agent's main processing loop."""
        self.logger.info(f"Starting {self.name}")
        
        # Start polling for telemetry data
        self.polling_task = asyncio.create_task(self._poll_telemetry())
        
        # Wait for polling task to complete
        try:
            await self.polling_task
        except asyncio.CancelledError:
            self.logger.info(f"Polling task cancelled")
        
    async def _shutdown(self) -> None:
        """Shut down the agent."""
        if self.polling_task and not self.polling_task.done():
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass
        
        if self.mcp_client:
            await self.mcp_client.close()
            
        if self.acp_client:
            await self.acp_client.close()
            
        self.logger.info(f"Shut down {self.name}")
        
    async def _poll_telemetry(self) -> None:
        """Poll for new telemetry data periodically."""
        while True:
            try:
                # Fetch telemetry data
                telemetry = await self._fetch_telemetry()
                
                if telemetry:
                    # Process telemetry data
                    anomalies = await self._process_telemetry(telemetry)
                    
                    # Report any anomalies
                    if anomalies:
                        await self._report_anomalies(anomalies, telemetry)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Error in telemetry polling: {str(e)}")
                
            # Wait for next polling interval
            await asyncio.sleep(self.poll_interval)
            
    async def _fetch_telemetry(self) -> Dict[str, Any]:
        """Fetch telemetry data from the telemetry API.
        
        Returns:
            The telemetry data
        """
        # In a real implementation, this would make an API call to get telemetry data
        # For this example, we'll generate some simulated telemetry data
        timestamp = datetime.utcnow()
        
        # Generate baseline values
        base_values = {
            "packet_loss": 0.01,  # 1% packet loss
            "latency": 50.0,      # 50ms latency
            "throughput": 950.0,   # 950 Mbps
            "jitter": 5.0,        # 5ms jitter
            "signal_strength": -70.0  # -70 dBm
        }
        
        # Add some random noise
        telemetry = {}
        for metric, base_value in base_values.items():
            # Add some normal noise
            noise = np.random.normal(0, 0.1 * abs(base_value))
            value = base_value + noise
            
            # Occasionally introduce an anomaly (1% chance)
            if np.random.random() < 0.01:
                value = base_value * (1.5 + np.random.random())
                
            telemetry[metric] = value
            
        # Add timestamp and metadata
        telemetry["timestamp"] = timestamp.isoformat()
        telemetry["cell_id"] = "SITE_123"
        telemetry["region"] = "REGION_A"
        
        self.logger.debug(f"Fetched telemetry data: {telemetry}")
        return telemetry
        
    async def _process_telemetry(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process telemetry data and detect anomalies.
        
        Args:
            telemetry: The telemetry data to process
            
        Returns:
            List of detected anomalies
        """
        timestamp = datetime.fromisoformat(telemetry["timestamp"])
        anomalies = []
        
        # Update metrics and check for anomalies
        for metric_name, metric in self.metrics.items():
            if metric_name in telemetry:
                value = telemetry[metric_name]
                metric.add_value(value, timestamp)
                
                # Check for anomaly
                is_anomaly, z_score, description = metric.detect_anomaly()
                if is_anomaly:
                    anomaly = {
                        "metric": metric_name,
                        "value": value,
                        "z_score": z_score,
                        "description": description,
                        "timestamp": timestamp.isoformat(),
                        "cell_id": telemetry.get("cell_id"),
                        "region": telemetry.get("region")
                    }
                    anomalies.append(anomaly)
                    self.logger.warning(f"Detected anomaly: {description}")
        
        return anomalies
        
    async def _report_anomalies(self, anomalies: List[Dict[str, Any]], telemetry: Dict[str, Any]) -> None:
        """Report detected anomalies.
        
        Args:
            anomalies: The detected anomalies
            telemetry: The telemetry data
        """
        # Create context in MCP
        context = self.mcp_client.create_context(
            domain_type=ContextType.NETWORK_DATA,
            payload={
                "anomalies": anomalies,
                "telemetry": telemetry,
                "metrics_statistics": {
                    name: metric.get_statistics() for name, metric in self.metrics.items()
                }
            },
            priority=3,  # Higher priority for anomalies
            confidence=ConfidenceLevel.HIGH,
            intent_tags=["anomaly_detection", "network_monitoring"],
            scope=telemetry.get("cell_id")
        )
        
        # Store context in MCP
        response = await self.mcp_client.send_context(context)
        if response and "context_id" in response:
            context_id = response["context_id"]
            self.logger.info(f"Stored anomaly context in MCP with ID {context_id}")
            
            # Notify planning agents
            await self._notify_planning_agents(anomalies, context_id)
        
    async def _notify_planning_agents(self, anomalies: List[Dict[str, Any]], context_id: str) -> None:
        """Notify planning agents about detected anomalies.
        
        Args:
            anomalies: The detected anomalies
            context_id: ID of the context in MCP
        """
        # For simplicity, we'll broadcast to all planning agents
        # In a real implementation, you might want to target specific agents
        message = self.acp_client.create_message(
            recipient_id=None,  # Broadcast
            message_type=MessageType.NOTIFICATION,
            action_type=ActionType.PLAN,
            payload={
                "anomalies": anomalies,
                "severity": "medium",
                "detection_time": datetime.utcnow().isoformat()
            },
            priority=MessagePriority.HIGH,
            context_refs=[context_id]
        )
        
        await self.acp_client.send_message(message)
        self.logger.info(f"Notified planning agents about anomalies")
        
    async def _handle_request(self, message: Dict[str, Any]) -> None:
        """Handle an incoming request message.
        
        Args:
            message: The request message
        """
        if message["action_type"] == ActionType.DIAGNOSE:
            # Handle diagnostic request
            payload = message["payload"]
            
            # Perform diagnostic based on request
            diagnostic_result = await self.perform_diagnostic(payload)
            
            # Send response
            response = self.acp_client.create_message(
                recipient_id=message["header"]["sender_id"],
                message_type=MessageType.RESPONSE,
                action_type=ActionType.DIAGNOSE,
                payload=diagnostic_result,
                priority=message["header"]["priority"],
                in_reply_to=message["header"]["message_id"],
                conversation_id=message["header"]["conversation_id"],
                context_refs=message["context_refs"]
            )
            
            await self.acp_client.send_message(response)
        
    async def perform_diagnostic(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a diagnostic based on the request.
        
        Args:
            request: The diagnostic request
            
        Returns:
            The diagnostic result
        """
        # In a real implementation, this would perform a specific diagnostic
        # For this example, we'll just return the current metrics statistics
        return {
            "metrics_statistics": {
                name: metric.get_statistics() for name, metric in self.metrics.items()
            },
            "diagnostic_time": datetime.utcnow().isoformat(),
            "diagnostic_agent": self.agent_id
        }
        
    async def detect_anomalies(self, telemetry_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in the provided telemetry data.
        
        Args:
            telemetry_data: The telemetry data to analyze
            
        Returns:
            List of detected anomalies with their details
        """
        return await self._process_telemetry(telemetry_data)
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        # This would be implemented for more complex message processing
        # For now, we'll just return a simple acknowledgement
        return {
            "status": "acknowledged",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities supported by this agent."""
        return [
            "network_anomaly_detection",
            "telemetry_analysis",
            "packet_loss_diagnostics",
            "latency_diagnostics",
            "throughput_diagnostics",
            "jitter_diagnostics",
            "signal_strength_diagnostics"
        ]
