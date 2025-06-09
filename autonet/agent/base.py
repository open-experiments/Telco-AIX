# Author: Fatih E. NAR
# Agentic AI Framework
#
import uuid
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class AgentState(Enum):
    """Possible states of an agent."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    TERMINATED = "terminated"
    ERROR = "error"

class AgentType(Enum):
    """Types of agents in the telecom agent framework."""
    DIAGNOSTIC = "diagnostic"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"

class Agent(ABC):
    """Base class for all agents in the telecom agent framework."""
    
    def __init__(self, agent_id: Optional[str] = None, agent_type: AgentType = None, 
                 name: str = None, description: str = None, config: Dict[str, Any] = None):
        """Initialize a new agent.
        
        Args:
            agent_id: Unique identifier for the agent, generated if not provided
            agent_type: Type of the agent
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            config: Configuration parameters for the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}-agent-{self.agent_id[:8]}"
        self.description = description or "Telecom agent"
        self.config = config or {}
        self.state = AgentState.INITIALIZING
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self._event_loop = None
        self._tasks = []
        
    async def initialize(self) -> None:
        """Initialize the agent and prepare it for operation."""
        self.logger.info(f"Initializing agent {self.name} ({self.agent_id})")
        try:
            await self._initialize()
            self.state = AgentState.IDLE
            self.logger.info(f"Agent {self.name} initialized successfully")
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to initialize agent {self.name}: {str(e)}")
            raise
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Implement agent-specific initialization logic."""
        pass
    
    async def start(self) -> None:
        """Start the agent's operation."""
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Agent {self.name} is not in IDLE state, current state: {self.state}")
        
        self.logger.info(f"Starting agent {self.name}")
        self.state = AgentState.PROCESSING
        
        # Get or create an event loop
        try:
            self._event_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        
        try:
            # Start the agent-specific processing
            task = self._event_loop.create_task(self._run())
            self._tasks.append(task)
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to start agent {self.name}: {str(e)}")
            raise
    
    @abstractmethod
    async def _run(self) -> None:
        """Implement the agent's main processing logic."""
        pass
    
    async def stop(self) -> None:
        """Stop the agent's operation gracefully."""
        self.logger.info(f"Stopping agent {self.name}")
        
        try:
            # Cancel all running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to be cancelled
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Run agent-specific shutdown logic
            await self._shutdown()
            
            self.state = AgentState.TERMINATED
            self.logger.info(f"Agent {self.name} stopped successfully")
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Error stopping agent {self.name}: {str(e)}")
            raise
    
    @abstractmethod
    async def _shutdown(self) -> None:
        """Implement agent-specific shutdown logic."""
        pass
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response.
        
        Args:
            message: The message to process
            
        Returns:
            The response message
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities supported by this agent."""
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Return the current status of the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type.value if self.agent_type else None,
            "state": self.state.value,
            "capabilities": self.get_capabilities()
        }

class DiagnosticAgent(Agent):
    """Base class for diagnostic agents that monitor and analyze network conditions."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None, 
                 description: str = None, config: Dict[str, Any] = None):
        """Initialize a new diagnostic agent."""
        super().__init__(agent_id, AgentType.DIAGNOSTIC, name, description, config)
        
    @abstractmethod
    async def detect_anomalies(self, telemetry_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in the provided telemetry data.
        
        Args:
            telemetry_data: The telemetry data to analyze
            
        Returns:
            List of detected anomalies with their details
        """
        pass

class PlanningAgent(Agent):
    """Base class for planning agents that generate solutions for detected issues."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None, 
                 description: str = None, config: Dict[str, Any] = None):
        """Initialize a new planning agent."""
        super().__init__(agent_id, AgentType.PLANNING, name, description, config)
        
    @abstractmethod
    async def generate_plan(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan to address the described problem.
        
        Args:
            problem_context: The context describing the problem to solve
            
        Returns:
            A plan with steps to address the problem
        """
        pass

class ExecutionAgent(Agent):
    """Base class for execution agents that implement solutions."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None, 
                 description: str = None, config: Dict[str, Any] = None):
        """Initialize a new execution agent."""
        super().__init__(agent_id, AgentType.EXECUTION, name, description, config)
        
    @abstractmethod
    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the provided plan.
        
        Args:
            plan: The plan to execute
            
        Returns:
            The result of the execution
        """
        pass

class ValidationAgent(Agent):
    """Base class for validation agents that verify solutions."""
    
    def __init__(self, agent_id: Optional[str] = None, name: str = None, 
                 description: str = None, config: Dict[str, Any] = None):
        """Initialize a new validation agent."""
        super().__init__(agent_id, AgentType.VALIDATION, name, description, config)
        
    @abstractmethod
    async def validate_execution(self, execution_result: Dict[str, Any], 
                               original_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the execution of a plan.
        
        Args:
            execution_result: The result of executing a plan
            original_plan: The original plan that was executed
            
        Returns:
            Validation results
        """
        pass
