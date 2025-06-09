# Author: Fatih E. NAR
# Agentic AI Framework
#
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import uuid

class MessageType(str, Enum):
    """Types of messages in Agent Communication Protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    DISCOVERY = "discovery"
    HEARTBEAT = "heartbeat"

class ActionType(str, Enum):
    """Types of actions agents can request from each other."""
    DIAGNOSE = "diagnose"
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    QUERY = "query"
    INFORM = "inform"

class MessagePriority(int, Enum):
    """Priority levels for ACP messages."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ROUTINE = 5

class ACPHeader(BaseModel):
    """Header information for ACP messages."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this message")
    sender_id: str = Field(..., description="Identifier of the sending agent")
    recipient_id: Optional[str] = Field(None, description="Identifier of the recipient agent (None for broadcasts)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the message was created")
    message_type: MessageType = Field(..., description="Type of message")
    priority: MessagePriority = Field(default=MessagePriority.MEDIUM, description="Priority level of the message")
    in_reply_to: Optional[str] = Field(None, description="ID of the message this is a reply to")
    conversation_id: Optional[str] = Field(None, description="ID to group related messages")

class CapabilityInfo(BaseModel):
    """Information about an agent's capabilities."""
    action_types: List[ActionType] = Field(..., description="Actions this agent can perform")
    domains: List[str] = Field(..., description="Domains this agent operates in")
    description: str = Field(..., description="Human-readable description of capabilities")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Any constraints on capabilities")

class DiscoveryData(BaseModel):
    """Data for discovery messages."""
    agent_type: str = Field(..., description="Type of the agent")
    capabilities: CapabilityInfo = Field(..., description="Agent's capabilities")
    network_location: Dict[str, Any] = Field(..., description="Network location information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ACPMessage(BaseModel):
    """A message in the Agent Communication Protocol."""
    header: ACPHeader
    action_type: Optional[ActionType] = Field(None, description="Type of action requested/performed")
    context_refs: List[str] = Field(default_factory=list, description="References to context in MCP")
    payload: Dict[str, Any] = Field(..., description="The actual message content")
    
    class Config:
        schema_extra = {
            "example": {
                "header": {
                    "message_id": "msg-1234",
                    "sender_id": "diagnostic-agent-1",
                    "recipient_id": "planning-agent-1",
                    "timestamp": "2023-04-15T09:30:00Z",
                    "message_type": "request",
                    "priority": 2,
                    "conversation_id": "incident-5678"
                },
                "action_type": "plan",
                "context_refs": ["ctx-9876"],
                "payload": {
                    "anomaly": {
                        "description": "High packet loss detected on cell site A123",
                        "severity": "medium",
                        "metrics": {
                            "packet_loss": 0.05,
                            "time_detected": "2023-04-15T09:25:00Z"
                        }
                    },
                    "request": {
                        "resolution_type": "automatic",
                        "time_sensitivity": "medium"
                    }
                }
            }
        }

class AgentDescription(BaseModel):
    """Description of an agent for discovery and coordination."""
    agent_id: str
    agent_type: str
    name: str
    capabilities: CapabilityInfo
    network_location: Dict[str, Any]
    status: str
    last_seen: datetime
