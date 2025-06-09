# Author: Fatih E. NAR
# Agentic AI Framework
#
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ContextType(str, Enum):
    """Types of context in Model Context Protocol."""
    NETWORK_DATA = "network_data"
    CUSTOMER_INFO = "customer_info"
    SERVICE_METRICS = "service_metrics"
    CONFIGURATION = "configuration"
    INCIDENT = "incident"
    HISTORICAL = "historical"
    PROCEDURAL = "procedural"
    REGULATORY = "regulatory"

class ConfidenceLevel(str, Enum):
    """Confidence levels for context information."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class MCPHeader(BaseModel):
    """Header information for MCP messages."""
    source_id: str = Field(..., description="Identifier of the source component")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when the context was created")
    priority: int = Field(default=5, ge=1, le=10, description="Priority level (1-10, where 1 is highest)")
    correlation_id: Optional[str] = Field(None, description="Correlation identifier for tracking related contexts")
    
class DomainMetadata(BaseModel):
    """Metadata about the domain context."""
    domain_type: ContextType
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    intent_tags: List[str] = Field(default_factory=list, description="Tags describing the intent or purpose")
    scope: Optional[str] = Field(None, description="Scope of the context (e.g., 'network_segment_X')")
    
class MCPContext(BaseModel):
    """A Model Context Protocol message containing context information."""
    header: MCPHeader
    metadata: DomainMetadata
    payload: Dict[str, Any] = Field(..., description="The actual context data")
    
    class Config:
        schema_extra = {
            "example": {
                "header": {
                    "source_id": "diagnostic-agent-1",
                    "timestamp": "2023-04-15T09:30:00Z",
                    "priority": 3,
                    "correlation_id": "incident-12345"
                },
                "metadata": {
                    "domain_type": "network_data",
                    "confidence": "high",
                    "intent_tags": ["anomaly_detection", "radio_network"],
                    "scope": "cell_site_A123"
                },
                "payload": {
                    "metrics": {
                        "packet_loss": 0.05,
                        "latency": 120,
                        "throughput": 950,
                        "connection_drops": 15
                    },
                    "location": {
                        "site_id": "A123",
                        "region": "northeast",
                        "coordinates": {
                            "lat": 40.7128,
                            "lon": -74.0060
                        }
                    },
                    "time_window": {
                        "start": "2023-04-15T09:00:00Z",
                        "end": "2023-04-15T09:30:00Z"
                    }
                }
            }
        }
