# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import json
from typing import Dict, List, Set, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import ValidationError

from .schema import ACPMessage

class ACPMessageBroker:
    """Broker for the Agent Communication Protocol (ACP)."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002):
        """Initialize the ACP broker.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port
        self.logger = logging.getLogger("acp.broker")
        self.app = FastAPI(title="Agent Communication Protocol Broker")
        
        # Connected clients
        self.connections: Dict[str, WebSocket] = {}
        
        # Message history (recent messages for recovery)
        self.message_history: Dict[str, List[Dict]] = {}
        self.history_size = 100  # How many messages to keep per recipient
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up the API routes."""
        
        @self.app.websocket("/ws/{agent_id}")
        async def websocket_endpoint(websocket: WebSocket, agent_id: str):
            self.logger.info(f"WebSocket connection request from agent {agent_id}")
            await self._handle_connection(websocket, agent_id)
            
        @self.app.get("/agents")
        async def get_agents():
            """Get a list of connected agents."""
            return {
                "status": "success",
                "agents": list(self.connections.keys()),
                "count": len(self.connections)
            }
            
        @self.app.post("/messages")
        async def send_message(message: dict):
            """Send a message via the broker's REST API."""
            try:
                # Validate and parse the message
                acp_message = ACPMessage.parse_obj(message)
                
                # Process and deliver the message
                await self._process_message(acp_message)
                
                return {"status": "success", "message_id": acp_message.header.message_id}
            except ValidationError as e:
                self.logger.error(f"Invalid message format: {str(e)}")
                return {"status": "error", "detail": f"Invalid message format: {str(e)}"}
            except Exception as e:
                self.logger.error(f"Error sending message: {str(e)}")
                return {"status": "error", "detail": str(e)}
        
    async def _handle_connection(self, websocket: WebSocket, agent_id: str):
        """Handle a WebSocket connection from an agent.
        
        Args:
            websocket: The WebSocket connection
            agent_id: ID of the connecting agent
        """
        try:
            # Accept the connection
            await websocket.accept()
            self.logger.info(f"WebSocket connection accepted for agent {agent_id}")
            
            # Register the connection
            self.connections[agent_id] = websocket
            self.logger.info(f"Agent {agent_id} connected")
            
            # Send any pending messages
            await self._send_pending_messages(agent_id)
            
            # Handle messages
            while True:
                # Receive message
                try:
                    data = await websocket.receive_text()
                    self.logger.debug(f"Received message from {agent_id}: {data[:100]}...")
                    
                    message_data = json.loads(data)
                    
                    try:
                        # Parse and validate the message
                        message = ACPMessage.parse_obj(message_data)
                        
                        # Process and deliver the message
                        await self._process_message(message)
                    except ValidationError as e:
                        self.logger.error(f"Invalid message from {agent_id}: {str(e)}")
                        await websocket.send_json({
                            "status": "error",
                            "detail": f"Invalid message format: {str(e)}"
                        })
                    except Exception as e:
                        self.logger.error(f"Error processing message from {agent_id}: {str(e)}")
                        await websocket.send_json({
                            "status": "error",
                            "detail": str(e)
                        })
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received from {agent_id}")
                    await websocket.send_json({
                        "status": "error",
                        "detail": "Invalid JSON format"
                    })
        except WebSocketDisconnect:
            self.logger.info(f"Agent {agent_id} disconnected")
            if agent_id in self.connections:
                del self.connections[agent_id]
        except Exception as e:
            self.logger.error(f"Error in WebSocket connection with {agent_id}: {str(e)}")
            if agent_id in self.connections:
                del self.connections[agent_id]
                
    async def _send_pending_messages(self, agent_id: str):
        """Send any pending messages to a newly connected agent.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.message_history and self.message_history[agent_id]:
            websocket = self.connections.get(agent_id)
            if websocket:
                self.logger.info(f"Sending {len(self.message_history[agent_id])} pending messages to {agent_id}")
                for message in self.message_history[agent_id]:
                    try:
                        await websocket.send_json(message)
                        self.logger.debug(f"Sent pending message to {agent_id}")
                    except Exception as e:
                        self.logger.error(f"Error sending pending message to {agent_id}: {str(e)}")
                        
    async def _process_message(self, message: ACPMessage):
        """Process and deliver a message.
        
        Args:
            message: The message to process
        """
        # Get message as dict for storage
        message_dict = message.dict()
        
        recipient_id = message.header.recipient_id
        
        if recipient_id:
            # Direct message to a specific recipient
            await self._deliver_message(message_dict, recipient_id)
        else:
            # Broadcast message
            await self._broadcast_message(message_dict)
            
    async def _deliver_message(self, message: Dict, recipient_id: str):
        """Deliver a message to a specific recipient.
        
        Args:
            message: The message to deliver
            recipient_id: ID of the recipient
        """
        # Store in history for recipient
        if recipient_id not in self.message_history:
            self.message_history[recipient_id] = []
            
        # Add to history, removing older messages if needed
        self.message_history[recipient_id].append(message)
        if len(self.message_history[recipient_id]) > self.history_size:
            self.message_history[recipient_id].pop(0)
        
        # If recipient is connected, deliver immediately
        if recipient_id in self.connections:
            websocket = self.connections[recipient_id]
            try:
                await websocket.send_json(message)
                self.logger.debug(f"Delivered message from {message['header']['sender_id']} to {recipient_id}")
            except Exception as e:
                self.logger.error(f"Error delivering message to {recipient_id}: {str(e)}")
        else:
            self.logger.debug(f"Recipient {recipient_id} not connected, message stored in history")
            
    async def _broadcast_message(self, message: Dict):
        """Broadcast a message to all connected agents except the sender.
        
        Args:
            message: The message to broadcast
        """
        sender_id = message["header"]["sender_id"]
        
        for agent_id, websocket in self.connections.items():
            if agent_id != sender_id:
                try:
                    await websocket.send_json(message)
                    self.logger.debug(f"Broadcast message from {sender_id} to {agent_id}")
                    
                    # Store in history for recipient
                    if agent_id not in self.message_history:
                        self.message_history[agent_id] = []
                        
                    # Add to history, removing older messages if needed
                    self.message_history[agent_id].append(message)
                    if len(self.message_history[agent_id]) > self.history_size:
                        self.message_history[agent_id].pop(0)
                except Exception as e:
                    self.logger.error(f"Error broadcasting message to {agent_id}: {str(e)}")
                    
    async def start(self):
        """Start the ACP broker."""
        import uvicorn
        
        self.logger.info(f"Starting ACP broker on {self.host}:{self.port}")
        server = uvicorn.Server(uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port,
            log_level="info"
        ))
        await server.serve()
