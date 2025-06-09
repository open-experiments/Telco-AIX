# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
import aiohttp
from datetime import datetime

from .schema import ACPMessage, ACPHeader, MessageType, ActionType, MessagePriority, CapabilityInfo

class ACPMessagingClient:
    """Client for sending and receiving messages using the Agent Communication Protocol (ACP)."""
    
    def __init__(self, agent_id: str, broker_url: str):
        """Initialize the ACP messaging client.
        
        Args:
            agent_id: Unique identifier for this agent
            broker_url: URL of the ACP message broker
        """
        self.agent_id = agent_id
        self.broker_url = broker_url
        self.logger = logging.getLogger(f"acp.messaging.{agent_id}")
        self.session = None
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.action_handlers: Dict[ActionType, List[Callable]] = {}
        self.websocket = None
        self.listening_task = None
        self.is_connected = False
        
    async def initialize(self):
        """Initialize the client and connect to the broker."""
        self.session = aiohttp.ClientSession()
        await self._connect_to_broker()
        self.logger.info(f"ACP client {self.agent_id} initialized")
        
    async def _connect_to_broker(self):
        """Connect to the ACP message broker via WebSocket."""
        max_retries = 3
        retry_delay = 2.0  # seconds
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Connecting to ACP broker at {self.broker_url}/ws/{self.agent_id} (attempt {attempt+1}/{max_retries})")
                self.websocket = await self.session.ws_connect(f"{self.broker_url}/ws/{self.agent_id}")
                self.is_connected = True
                self.listening_task = asyncio.create_task(self._listen_for_messages())
                self.logger.info(f"Connected to ACP broker at {self.broker_url}")
                return
            except aiohttp.ClientError as e:
                self.is_connected = False
                self.logger.error(f"Failed to connect to ACP broker (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"Maximum connection attempts reached. Could not connect to broker.")
                    raise
        
    async def _listen_for_messages(self):
        """Listen for incoming messages from the broker."""
        try:
            async for msg in self.websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        message_data = msg.json()
                        message = ACPMessage.parse_obj(message_data)
                        await self._process_message(message)
                    except Exception as e:
                        self.logger.error(f"Error processing message: {str(e)}")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.info("WebSocket connection closed")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {self.websocket.exception()}")
                    break
        except asyncio.CancelledError:
            self.logger.info("Message listening task cancelled")
        except Exception as e:
            self.logger.error(f"Error in message listening loop: {str(e)}")
        finally:
            self.is_connected = False
            
    async def _process_message(self, message: ACPMessage):
        """Process an incoming ACP message.
        
        Args:
            message: The received ACP message
        """
        # Log message receipt
        self.logger.debug(f"Received message {message.header.message_id} from {message.header.sender_id}")
        
        # Handle based on message type
        if message.header.message_type in self.message_handlers:
            for handler in self.message_handlers[message.header.message_type]:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {str(e)}")
        
        # Handle based on action type
        if message.action_type and message.action_type in self.action_handlers:
            for handler in self.action_handlers[message.action_type]:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Error in action handler: {str(e)}")
    
    async def close(self):
        """Close the client and release resources."""
        if self.listening_task:
            self.listening_task.cancel()
            try:
                await self.listening_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            
        if self.session:
            await self.session.close()
            self.session = None
            
        self.is_connected = False
        self.logger.info(f"ACP client {self.agent_id} closed")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Callback function to handle the message
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        self.logger.info(f"Registered handler for message type {message_type}")
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """Register a handler for a specific action type.
        
        Args:
            action_type: Type of action to handle
            handler: Callback function to handle the action
        """
        if action_type not in self.action_handlers:
            self.action_handlers[action_type] = []
        
        self.action_handlers[action_type].append(handler)
        self.logger.info(f"Registered handler for action type {action_type}")
    
    def create_message(self, 
                     recipient_id: Optional[str],
                     message_type: MessageType,
                     action_type: Optional[ActionType],
                     payload: Dict[str, Any],
                     priority: MessagePriority = MessagePriority.MEDIUM,
                     context_refs: Optional[List[str]] = None,
                     in_reply_to: Optional[str] = None,
                     conversation_id: Optional[str] = None) -> ACPMessage:
        """Create a new ACP message.
        
        Args:
            recipient_id: ID of the recipient agent (None for broadcasts)
            message_type: Type of message
            action_type: Type of action requested/performed
            payload: The message payload
            priority: Message priority
            context_refs: References to MCP contexts
            in_reply_to: ID of the message this is replying to
            conversation_id: ID to group related messages
            
        Returns:
            A new ACPMessage object
        """
        header = ACPHeader(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            timestamp=datetime.utcnow(),
            message_type=message_type,
            priority=priority,
            in_reply_to=in_reply_to,
            conversation_id=conversation_id or str(uuid.uuid4())
        )
        
        return ACPMessage(
            header=header,
            action_type=action_type,
            context_refs=context_refs or [],
            payload=payload
        )
    
    async def send_message(self, message: ACPMessage) -> bool:
        """Send an ACP message to the broker.
        
        Args:
            message: The message to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        if not self.is_connected:
            await self._connect_to_broker()
            
        try:
            await self.websocket.send_json(message.dict())
            self.logger.debug(f"Sent message {message.header.message_id} to {message.header.recipient_id or 'broadcast'}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return False
    
    async def request_response(self, 
                             recipient_id: str,
                             action_type: ActionType,
                             payload: Dict[str, Any],
                             priority: MessagePriority = MessagePriority.MEDIUM,
                             context_refs: Optional[List[str]] = None,
                             conversation_id: Optional[str] = None,
                             timeout: float = 30.0) -> Optional[ACPMessage]:
        """Send a request message and wait for a response.
        
        Args:
            recipient_id: ID of the recipient agent
            action_type: Type of action requested
            payload: The request payload
            priority: Message priority
            context_refs: References to MCP contexts
            conversation_id: ID to group related messages
            timeout: Maximum time to wait for a response (seconds)
            
        Returns:
            The response message, or None if no response was received within the timeout
        """
        # Create and send request message
        request = self.create_message(
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            action_type=action_type,
            payload=payload,
            priority=priority,
            context_refs=context_refs,
            conversation_id=conversation_id
        )
        
        # Set up a future to receive the response
        response_future = asyncio.Future()
        
        # Handler for response messages
        async def response_handler(message: ACPMessage):
            if (message.header.in_reply_to == request.header.message_id and 
                message.header.sender_id == recipient_id):
                response_future.set_result(message)
        
        # Register temporary handler
        self.register_message_handler(MessageType.RESPONSE, response_handler)
        
        try:
            # Send the request
            if not await self.send_message(request):
                return None
            
            # Wait for response with timeout
            return await asyncio.wait_for(response_future, timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"No response received from {recipient_id} within {timeout} seconds")
            return None
        finally:
            # Remove temporary handler
            if MessageType.RESPONSE in self.message_handlers:
                self.message_handlers[MessageType.RESPONSE].remove(response_handler)
