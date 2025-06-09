# Author: Fatih E. NAR
# Agentic AI Framework
#
import asyncio
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
from datetime import datetime

class DashboardServer:
    """Server for the web dashboard of the Telecom AI Agent Framework."""
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 8080,
                 mcp_url: str = "http://localhost:8000",
                 acp_url: str = "http://localhost:8002",
                 orchestration_service = None):
        """Initialize the dashboard server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            mcp_url: URL for the MCP server
            acp_url: URL for the ACP broker
            orchestration_service: Reference to the orchestration service
        """
        self.host = host
        self.port = port
        self.mcp_url = mcp_url
        self.acp_url = acp_url
        self.orchestration_service = orchestration_service
        self.logger = logging.getLogger("dashboard.server")
        
        # Create FastAPI app
        self.app = FastAPI(title="Telecom AI Agent Framework Dashboard")
        
        # Connected dashboard clients
        self.websocket_clients: List[WebSocket] = []
        
        # Setup routes and static files
        self._setup_routes()
        
    def _setup_routes(self):
        """Set up the API routes and static files."""
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up static files
        static_dir = os.path.join(current_dir, "static")
        os.makedirs(static_dir, exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Set up templates
        templates_dir = os.path.join(current_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        self.templates = Jinja2Templates(directory=templates_dir)
        
        # Main dashboard page
        @self.app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        # Agent details page
        @self.app.get("/agents/{agent_id}", response_class=HTMLResponse)
        async def agent_details(request: Request, agent_id: str):
            return self.templates.TemplateResponse("agent.html", {
                "request": request, 
                "agent_id": agent_id
            })
        
        # Workflow details page
        @self.app.get("/workflows/{workflow_id}", response_class=HTMLResponse)
        async def workflow_details(request: Request, workflow_id: str):
            return self.templates.TemplateResponse("workflow.html", {
                "request": request, 
                "workflow_id": workflow_id
            })
        
        # API to get agents
        @self.app.get("/api/agents")
        async def get_agents():
            if self.orchestration_service:
                return {
                    "status": "success",
                    "agents": [agent.dict() for agent in self.orchestration_service.agent_registry.values()]
                }
            return {"status": "error", "message": "Orchestration service not available"}
        
        # API to get specific agent
        @self.app.get("/api/agents/{agent_id}")
        async def get_agent(agent_id: str):
            if self.orchestration_service:
                agent = self.orchestration_service.agent_registry.get(agent_id)
                if agent:
                    return {"status": "success", "agent": agent.dict()}
                return {"status": "error", "message": f"Agent {agent_id} not found"}
            return {"status": "error", "message": "Orchestration service not available"}
        
        # API to get agent messages (placeholder for now)
        @self.app.get("/api/agents/{agent_id}/messages")
        async def get_agent_messages(agent_id: str):
            if self.orchestration_service:
                agent = self.orchestration_service.agent_registry.get(agent_id)
                if agent:
                    # For now, return placeholder messages
                    # In a real implementation, this would fetch actual message history
                    messages = [
                        {
                            "type": "INFO",
                            "content": f"Agent {agent.name} initialized successfully",
                            "timestamp": agent.last_seen.isoformat()
                        },
                        {
                            "type": "STATUS",
                            "content": f"Agent status: {agent.status}",
                            "timestamp": agent.last_seen.isoformat()
                        }
                    ]
                    return {"status": "success", "messages": messages}
                return {"status": "error", "message": f"Agent {agent_id} not found"}
            return {"status": "error", "message": "Orchestration service not available"}
        
        # API to get workflows
        @self.app.get("/api/workflows")
        async def get_workflows():
            if self.orchestration_service:
                return {
                    "status": "success",
                    "workflows": self.orchestration_service.workflows
                }
            return {"status": "error", "message": "Orchestration service not available"}
        
        # API to get workflow status
        @self.app.get("/api/workflows/{workflow_id}")
        async def get_workflow_status(workflow_id: str):
            if self.orchestration_service:
                status = await self.orchestration_service.get_workflow_status(workflow_id)
                if status:
                    return {"status": "success", "workflow": status}
                return {"status": "error", "message": f"Workflow {workflow_id} not found"}
            return {"status": "error", "message": "Orchestration service not available"}
        
        # WebSocket for real-time updates
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_clients.append(websocket)
            try:
                while True:
                    # Wait for messages from client (not used now, just keep connection alive)
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                # Safely remove websocket if it exists in the list
                if websocket in self.websocket_clients:
                    self.websocket_clients.remove(websocket)
            except Exception as e:
                # Handle any other exceptions and safely remove websocket
                if websocket in self.websocket_clients:
                    self.websocket_clients.remove(websocket)
        
    async def broadcast_update(self, event_type: str, data: Any):
        """Broadcast an update to all connected dashboard clients.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.websocket_clients:
            return
            
        message = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected_clients.append(client)
                
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.websocket_clients:
                self.websocket_clients.remove(client)
                
    async def start_update_task(self):
        """Start a task to periodically broadcast updates."""
        while True:
            try:
                if self.orchestration_service:
                    # Send agent updates
                    agents = [agent.dict() for agent in self.orchestration_service.agent_registry.values()]
                    await self.broadcast_update("agents_update", agents)
                    
                    # Send workflow updates
                    workflows = []
                    for workflow_id in self.orchestration_service.workflows:
                        status = await self.orchestration_service.get_workflow_status(workflow_id)
                        if status:
                            workflows.append(status)
                    
                    await self.broadcast_update("workflows_update", workflows)
            except Exception as e:
                self.logger.error(f"Error in update task: {str(e)}")
                
            # Wait before next update
            await asyncio.sleep(2.0)
                
    async def start(self):
        """Start the dashboard server."""
        import uvicorn
        
        # Start update task
        update_task = asyncio.create_task(self.start_update_task())
        
        # Start web server
        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        server = uvicorn.Server(uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port,
            log_level="info"
        ))
        
        try:
            await server.serve()
        finally:
            # Cancel update task
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass
