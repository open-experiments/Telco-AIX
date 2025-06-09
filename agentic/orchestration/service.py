# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

from protocols.acp.schema import (
    ACPMessage, AgentDescription, MessageType, ActionType, CapabilityInfo
)

class OrchestrationService:
    """Service for orchestrating agent interactions.
    
    The orchestration service manages agent discovery, task decomposition, 
    workflow management, and resource allocation.
    """
    
    def __init__(self, service_id: Optional[str] = None):
        """Initialize a new orchestration service.
        
        Args:
            service_id: Unique identifier for this service instance
        """
        self.service_id = service_id or f"orchestrator-{str(uuid.uuid4())[:8]}"
        self.logger = logging.getLogger(f"orchestration.{self.service_id}")
        
        # Registry of known agents
        self.agent_registry: Dict[str, AgentDescription] = {}
        
        # Active workflows
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
        # Event subscribers
        self.event_subscribers: Dict[str, Set[str]] = {}
        
        # Running tasks
        self._tasks = []
        
    async def initialize(self) -> None:
        """Initialize the orchestration service."""
        self.logger.info(f"Initializing orchestration service {self.service_id}")
        
    async def shutdown(self) -> None:
        """Shut down the orchestration service."""
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to be cancelled
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        self.logger.info(f"Shut down orchestration service {self.service_id}")
        
    async def register_agent(self, agent_description: AgentDescription) -> None:
        """Register an agent with the orchestration service.
        
        Args:
            agent_description: Description of the agent to register
        """
        self.agent_registry[agent_description.agent_id] = agent_description
        self.logger.info(f"Registered agent {agent_description.agent_id} ({agent_description.agent_type})")
        
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestration service.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
            
    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update the status of an agent.
        
        Args:
            agent_id: ID of the agent
            status: New status
        """
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id].status = status
            self.agent_registry[agent_id].last_seen = datetime.now(timezone.utc)
            self.logger.debug(f"Updated status of agent {agent_id} to {status}")
            
    async def find_agents_by_type(self, agent_type: str) -> List[AgentDescription]:
        """Find agents of a specific type.
        
        Args:
            agent_type: Type of agent to find
            
        Returns:
            List of matching agent descriptions
        """
        return [
            agent for agent in self.agent_registry.values()
            if agent.agent_type == agent_type
        ]
        
    async def find_agents_by_capability(self, capability: str) -> List[AgentDescription]:
        """Find agents with a specific capability.
        
        Args:
            capability: Capability to look for
            
        Returns:
            List of matching agent descriptions
        """
        return [
            agent for agent in self.agent_registry.values()
            if capability in agent.capabilities.action_types
        ]
        
    async def create_workflow(self, 
                            workflow_type: str, 
                            parameters: Dict[str, Any],
                            initiator_id: Optional[str] = None) -> str:
        """Create a new workflow.
        
        Args:
            workflow_type: Type of workflow to create
            parameters: Workflow parameters
            initiator_id: ID of the agent initiating the workflow
            
        Returns:
            ID of the created workflow
        """
        workflow_id = str(uuid.uuid4())
        
        workflow = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "parameters": parameters,
            "initiator_id": initiator_id,
            "status": "created",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "steps": [],
            "results": {}
        }
        
        self.workflows[workflow_id] = workflow
        self.logger.info(f"Created workflow {workflow_id} of type {workflow_type}")
        
        # Start workflow execution
        task = asyncio.create_task(self._execute_workflow(workflow_id))
        self._tasks.append(task)
        
        return workflow_id
        
    async def _execute_workflow(self, workflow_id: str) -> None:
        """Execute a workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
        """
        if workflow_id not in self.workflows:
            self.logger.error(f"Workflow {workflow_id} not found")
            return
            
        workflow = self.workflows[workflow_id]
        workflow_type = workflow["workflow_type"]
        
        self.logger.info(f"Executing workflow {workflow_id} of type {workflow_type}")
        
        try:
            # Update workflow status
            workflow["status"] = "running"
            workflow["updated_at"] = datetime.now(timezone.utc)
            
            # Execute workflow based on type
            if workflow_type == "anomaly_resolution":
                await self._execute_anomaly_resolution_workflow(workflow)
            elif workflow_type == "network_optimization":
                await self._execute_network_optimization_workflow(workflow)
            elif workflow_type == "customer_issue_resolution":
                await self._execute_customer_issue_workflow(workflow)
            else:
                self.logger.error(f"Unknown workflow type: {workflow_type}")
                workflow["status"] = "failed"
                workflow["error"] = f"Unknown workflow type: {workflow_type}"
                
            # Update workflow status if not already failed
            if workflow["status"] != "failed":
                workflow["status"] = "completed"
                workflow["completed_at"] = datetime.now(timezone.utc)
                
        except Exception as e:
            self.logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            workflow["status"] = "failed"
            workflow["error"] = str(e)
        finally:
            workflow["updated_at"] = datetime.now(timezone.utc)
            
    async def _execute_anomaly_resolution_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute an anomaly resolution workflow.
        
        Args:
            workflow: The workflow to execute
        """
        # This would implement the steps for resolving network anomalies
        # For this example, we'll define a simple workflow
        
        # Step 1: Find diagnostic agents
        diagnostic_agents = await self.find_agents_by_type("diagnostic")
        if not diagnostic_agents:
            raise RuntimeError("No diagnostic agents available")
            
        # Step 2: Get detailed diagnostics
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "diagnostics",
            "description": "Performing network diagnostics and anomaly detection",
            "status": "completed",
            "started_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "agents": [agent.agent_id for agent in diagnostic_agents]
        })
        
        # Step 3: Find planning agents
        planning_agents = await self.find_agents_by_type("planning")
        if not planning_agents:
            raise RuntimeError("No planning agents available")
            
        # Step 4: Generate resolution plan
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "planning",
            "description": "Generating remediation plan based on diagnostic results",
            "status": "completed",
            "started_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "agents": [agent.agent_id for agent in planning_agents]
        })
        
        # Step 5: Find execution agents
        execution_agents = await self.find_agents_by_type("execution")
        if not execution_agents:
            raise RuntimeError("No execution agents available")
            
        # Step 6: Execute resolution plan
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "execution",
            "description": "Executing the remediation plan on network elements",
            "status": "completed",
            "started_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "agents": [agent.agent_id for agent in execution_agents]
        })
        
        # Step 7: Find validation agents
        validation_agents = await self.find_agents_by_type("validation")
        if not validation_agents:
            raise RuntimeError("No validation agents available")
            
        # Step 8: Validate resolution
        # (In a real implementation, this would communicate with the agents)
        workflow["steps"].append({
            "step_id": "validation",
            "description": "Validating that the network issue has been resolved",
            "status": "completed",
            "started_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "agents": [agent.agent_id for agent in validation_agents]
        })
        
        # Record results
        workflow["results"] = {
            "resolution_status": "success",
            "resolution_time": (datetime.now(timezone.utc) - workflow["created_at"]).total_seconds(),
            "agents_involved": len(diagnostic_agents) + len(planning_agents) + len(execution_agents) + len(validation_agents)
        }
        
    async def _execute_network_optimization_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute a network optimization workflow.
        
        Args:
            workflow: The workflow to execute
        """
        # This would implement the steps for network optimization
        # Similar structure to the anomaly resolution workflow
        workflow["results"] = {
            "optimization_status": "success",
            "optimization_gain": "15%",
            "completion_time": datetime.now(timezone.utc).isoformat()
        }
        
    async def _execute_customer_issue_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute a customer issue resolution workflow.
        
        Args:
            workflow: The workflow to execute
        """
        # This would implement the steps for customer issue resolution
        # Similar structure to the anomaly resolution workflow
        workflow["results"] = {
            "resolution_status": "success",
            "customer_id": workflow["parameters"].get("customer_id"),
            "resolution_time": (datetime.now(timezone.utc) - workflow["created_at"]).total_seconds()
        }
        
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            Workflow status information, or None if not found
        """
        if workflow_id not in self.workflows:
            return None
            
        workflow = self.workflows[workflow_id]
        
        # Convert datetime fields in steps to ISO format
        steps_formatted = []
        for step in workflow["steps"]:
            step_formatted = step.copy()
            for time_field in ["started_at", "completed_at"]:
                if time_field in step_formatted and step_formatted[time_field]:
                    step_formatted[time_field] = step_formatted[time_field].isoformat()
            steps_formatted.append(step_formatted)
        
        return {
            "workflow_id": workflow["workflow_id"],
            "workflow_type": workflow["workflow_type"],
            "status": workflow["status"],
            "created_at": workflow["created_at"].isoformat(),
            "updated_at": workflow["updated_at"].isoformat(),
            "initiator_id": workflow.get("initiator_id"),
            "parameters": workflow.get("parameters", {}),
            "steps": steps_formatted,
            "steps_completed": len([step for step in workflow["steps"] if step["status"] == "completed"]),
            "steps_total": len(workflow["steps"]),
            "results": workflow.get("results", {})
        }
        
    async def subscribe_to_events(self, event_type: str, subscriber_id: str) -> None:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            subscriber_id: ID of the subscribing agent
        """
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = set()
            
        self.event_subscribers[event_type].add(subscriber_id)
        self.logger.info(f"Agent {subscriber_id} subscribed to {event_type} events")
        
    async def unsubscribe_from_events(self, event_type: str, subscriber_id: str) -> None:
        """Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of event to unsubscribe from
            subscriber_id: ID of the subscribing agent
        """
        if event_type in self.event_subscribers and subscriber_id in self.event_subscribers[event_type]:
            self.event_subscribers[event_type].remove(subscriber_id)
            self.logger.info(f"Agent {subscriber_id} unsubscribed from {event_type} events")
            
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish an event to subscribers.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type not in self.event_subscribers:
            return
            
        subscribers = self.event_subscribers[event_type]
        self.logger.info(f"Publishing {event_type} event to {len(subscribers)} subscribers")
        
        # In a real implementation, this would send messages to the subscribers
        # For this example, we'll just log the event
        event_data["published_at"] = datetime.now(timezone.utc).isoformat()
        event_data["subscribers"] = list(subscribers)
        
        self.logger.info(f"Published {event_type} event: {event_data}")
