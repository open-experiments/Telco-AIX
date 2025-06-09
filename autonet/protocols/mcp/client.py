# Author: Fatih E. NAR
# Agentic AI Framework
#
import aiohttp
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .schema import MCPContext, MCPHeader, DomainMetadata, ContextType, ConfidenceLevel

class MCPClient:
    """Client for sending and receiving MCP (Model Context Protocol) messages."""
    
    def __init__(self, client_id: str, base_url: str):
        """Initialize the MCP client.
        
        Args:
            client_id: Unique identifier for this client
            base_url: Base URL of the MCP server
        """
        self.client_id = client_id
        self.base_url = base_url
        self.logger = logging.getLogger(f"mcp.client.{client_id}")
        self.session = None
        
    async def __aenter__(self):
        """Initialize the aiohttp session for async context manager."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the aiohttp session for async context manager."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def initialize(self):
        """Initialize the client."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        self.logger.info(f"MCP client {self.client_id} initialized")
    
    async def close(self):
        """Close the client and release resources."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info(f"MCP client {self.client_id} closed")
    
    def create_context(self, 
                     domain_type: ContextType, 
                     payload: Dict[str, Any],
                     priority: int = 5,
                     confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                     correlation_id: Optional[str] = None,
                     intent_tags: Optional[list] = None,
                     scope: Optional[str] = None) -> MCPContext:
        """Create a new MCP context object.
        
        Args:
            domain_type: Type of context being created
            payload: The actual context data
            priority: Priority level (1-10, where 1 is highest)
            confidence: Confidence level for the context
            correlation_id: Optional correlation ID for related contexts
            intent_tags: Optional tags describing intent
            scope: Optional scope identifier
            
        Returns:
            A new MCPContext object
        """
        header = MCPHeader(
            source_id=self.client_id,
            timestamp=datetime.utcnow(),
            priority=priority,
            correlation_id=correlation_id
        )
        
        metadata = DomainMetadata(
            domain_type=domain_type,
            confidence=confidence,
            intent_tags=intent_tags or [],
            scope=scope
        )
        
        return MCPContext(
            header=header,
            metadata=metadata,
            payload=payload
        )
    
    async def send_context(self, context: MCPContext, endpoint: str = "/context") -> Dict[str, Any]:
        """Send context to the MCP server.
        
        Args:
            context: The context to send
            endpoint: API endpoint to send the context to
            
        Returns:
            The server response
        """
        if not self.session:
            await self.initialize()
            
        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Sending context to {url}")
        
        try:
            async with self.session.post(url, json=context.dict()) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Error sending context: {str(e)}")
            raise
    
    async def query_context(self, query: Dict[str, Any], endpoint: str = "/context/query") -> Dict[str, Any]:
        """Query context from the MCP server.
        
        Args:
            query: The query parameters
            endpoint: API endpoint for querying context
            
        Returns:
            The contexts matching the query
        """
        if not self.session:
            await self.initialize()
            
        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Querying context from {url}")
        
        try:
            async with self.session.post(url, json=query) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            self.logger.error(f"Error querying context: {str(e)}")
            raise
