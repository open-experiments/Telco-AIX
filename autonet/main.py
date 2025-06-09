# main_new3.py - Modular Enhanced NOC Dashboard with Real-time Metrics

import asyncio
import logging
import argparse
import sys
import os
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import subprocess
import shutil
from aiohttp import web, web_response
import aiohttp_cors
import random
import math
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("enhanced_noc")

class AnomalyTriggerType(Enum):
    NEW_ANOMALY = "new_anomaly"
    THRESHOLD_BREACH = "threshold_breach"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    MANUAL_TRIGGER = "manual_trigger"
    AUTONOMOUS_PROCESSING = "autonomous_processing"

class WorkflowStatus(Enum):
    QUEUED = "queued"
    DIAGNOSTIC = "diagnostic"
    PLANNING = "planning" 
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowStepStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class AnomalyEvent:
    """Represents an anomaly event from NOC data"""
    event_id: str
    trigger_type: AnomalyTriggerType
    component: str  # AMF, SMF, UPF
    timestamp: datetime
    anomaly_data: Dict[str, Any]
    recommendations: Optional[str] = None
    rca_result: Optional[str] = None
    severity: str = "MEDIUM"

@dataclass
class AgentInteraction:
    """Represents an interaction between agents"""
    from_agent: str
    to_agent: str
    message_type: str
    timestamp: datetime
    status: str = "active"  # active, completed, failed

@dataclass
class WorkflowStep:
    """Individual step in workflow execution"""
    step_id: str
    agent: str  # diagnostic, planning, execution, validation
    description: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    output: Optional[str] = None
    error: Optional[str] = None

@dataclass
class WorkflowExecution:
    """Complete workflow execution tracking"""
    workflow_id: str
    anomaly_event: AnomalyEvent  # Reference to triggering event
    status: WorkflowStatus = WorkflowStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    
    # Workflow steps
    steps: List[WorkflowStep] = field(default_factory=list)
    current_step_index: int = 0
    
    # Execution details - ENHANCED with playbook info
    selected_playbook: Optional[str] = None
    playbook_description: Optional[str] = None  # NEW: Human-readable description
    playbook_execution: Optional[Dict[str, Any]] = None
    success_rate: float = 0.0
    
    # Agent interactions
    agent_interactions: List[AgentInteraction] = field(default_factory=list)

# NEW: Enhanced Agent Topology Classes
@dataclass
class AgentMetrics:
    """Real-time agent performance metrics"""
    agent_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    task_queue_size: int = 0
    tasks_processed_per_minute: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    health_score: float = 100.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Historical data (last 60 measurements)
    cpu_history: deque = field(default_factory=lambda: deque(maxlen=60))
    memory_history: deque = field(default_factory=lambda: deque(maxlen=60))
    response_time_history: deque = field(default_factory=lambda: deque(maxlen=60))

@dataclass
class AgentCommunication:
    """Agent-to-agent communication tracking"""
    from_agent: str
    to_agent: str
    message_type: str
    data_size: int  # bytes
    latency_ms: float
    timestamp: datetime
    status: str = "success"  # success, failed, timeout
    retry_count: int = 0

@dataclass
class WorkflowDecision:
    """AI decision making tracking"""
    agent_id: str
    decision_type: str  # route, escalate, retry, abort
    confidence_score: float
    reasoning: str
    alternatives_considered: List[str]
    timestamp: datetime

# NEW: Real-time metrics tracking
@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    anomaly: bool = False
    anomaly_severity: Optional[str] = None

class MetricsTracker:
    """Tracks real-time metrics for visualization"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics = {
            'amf': {
                'registration_rate': deque(maxlen=history_size),
                'registration_success_rate': deque(maxlen=history_size),
                'auth_success_rate': deque(maxlen=history_size),
                'anomalies': []
            },
            'smf': {
                'session_est_rate': deque(maxlen=history_size),
                'session_success_rate': deque(maxlen=history_size),
                'ip_pool_usage': deque(maxlen=history_size),
                'anomalies': []
            },
            'upf': {
                'active_sessions': deque(maxlen=history_size),
                'throughput': deque(maxlen=history_size),
                'latency': deque(maxlen=history_size),
                'anomalies': []
            }
        }
        self.last_update = datetime.now()
    
    def add_metric_point(self, component: str, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a metric data point"""
        if timestamp is None:
            timestamp = datetime.now()
        
        component_lower = component.lower()
        if component_lower in self.metrics and metric_name in self.metrics[component_lower]:
            self.metrics[component_lower][metric_name].append(MetricPoint(timestamp, value))
    
    def add_anomaly(self, component: str, timestamp: Any, severity: str):
        """Record an anomaly occurrence"""
        component_lower = component.lower()
        if component_lower in self.metrics:
            # Convert timestamp to datetime if needed
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
                
            self.metrics[component_lower]['anomalies'].append({
                'timestamp': timestamp.isoformat(),
                'severity': severity
            })
    
    def generate_simulated_metrics(self, current_time: Optional[datetime] = None):
        """Generate simulated metrics for demo purposes"""
        if current_time is None:
            current_time = datetime.now()
        
        # AMF Metrics
        self.add_metric_point('amf', 'registration_rate', 85 + random.uniform(-5, 10), current_time)
        self.add_metric_point('amf', 'registration_success_rate', 92 + random.uniform(-3, 5), current_time)
        self.add_metric_point('amf', 'auth_success_rate', 88 + random.uniform(-4, 8), current_time)
        
        # SMF Metrics
        self.add_metric_point('smf', 'session_est_rate', 78 + random.uniform(-6, 12), current_time)
        self.add_metric_point('smf', 'session_success_rate', 89 + random.uniform(-4, 8), current_time)
        self.add_metric_point('smf', 'ip_pool_usage', 45 + random.uniform(-10, 20), current_time)
        
        # UPF Metrics
        self.add_metric_point('upf', 'active_sessions', 3500 + random.uniform(-200, 500), current_time)
        self.add_metric_point('upf', 'throughput', 850 + random.uniform(-100, 150), current_time)
        self.add_metric_point('upf', 'latency', 15 + random.uniform(-5, 10), current_time)
        
        self.last_update = current_time
    
    def get_metrics_data(self, window_size: int = 30) -> Dict[str, Any]:
        """Get formatted metrics data for API response"""
        result = {}
        
        for component in ['amf', 'smf', 'upf']:
            component_data = {
                'timestamps': [],
                'metrics': {},
                'anomalies': []
            }
            
            # Get recent anomalies (serialize timestamps)
            raw_anomalies = self.metrics[component]['anomalies'][-10:]  # Last 10 anomalies
            for anomaly in raw_anomalies:
                anomaly_copy = dict(anomaly)
                # Ensure timestamp is serialized
                if 'timestamp' in anomaly_copy and not isinstance(anomaly_copy['timestamp'], str):
                    if hasattr(anomaly_copy['timestamp'], 'isoformat'):
                        anomaly_copy['timestamp'] = anomaly_copy['timestamp'].isoformat()
                    else:
                        anomaly_copy['timestamp'] = str(anomaly_copy['timestamp'])
                component_data['anomalies'].append(anomaly_copy)
            
            # Get metric names (exclude 'anomalies')
            metric_names = [k for k in self.metrics[component].keys() if k != 'anomalies']
            
            # Get the last 'window_size' data points
            for metric_name in metric_names:
                metric_deque = self.metrics[component][metric_name]
                recent_points = list(metric_deque)[-window_size:]
                
                if recent_points:
                    # Extract timestamps (use the first metric's timestamps as reference)
                    if not component_data['timestamps']:
                        component_data['timestamps'] = [p.timestamp.isoformat() for p in recent_points]
                    
                    # Extract values
                    component_data['metrics'][metric_name] = [p.value for p in recent_points]
            
            result[component] = component_data
        
        return result

class EnhancedAgentTopologyManager:
    """Advanced agent topology management with real-time analytics"""
    
    def __init__(self):
        self.agent_metrics = {}
        self.communication_log = deque(maxlen=1000)
        self.decision_log = deque(maxlen=500)
        self.agent_dependencies = {
            'diagnostic': ['planning'],
            'planning': ['execution', 'diagnostic'],  # can escalate back
            'execution': ['validation', 'planning'],  # can request replanning
            'validation': ['diagnostic']  # can trigger re-analysis
        }
        self.workflow_patterns = {}
        self.bottleneck_predictions = {}
        
        # Initialize metrics for each agent
        for agent_id in ['diagnostic', 'planning', 'execution', 'validation']:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
    
    def update_agent_metrics(self, agent_id: str, workflow_active: bool = False):
        """Simulate realistic agent performance metrics"""
        metrics = self.agent_metrics[agent_id]
        
        # Simulate realistic metrics based on agent type and current load
        base_cpu = {'diagnostic': 30, 'planning': 45, 'execution': 60, 'validation': 25}
        base_memory = {'diagnostic': 40, 'planning': 35, 'execution': 55, 'validation': 30}
        
        # Add variance and load-based increases
        load_multiplier = 1.5 if workflow_active else 1.0
        metrics.cpu_usage = min(95, base_cpu[agent_id] * load_multiplier + random.uniform(-10, 15))
        metrics.memory_usage = min(90, base_memory[agent_id] * load_multiplier + random.uniform(-8, 12))
        
        # Task queue simulation
        if workflow_active:
            metrics.task_queue_size = random.randint(2, 8)
            metrics.tasks_processed_per_minute = random.uniform(15, 35)
            metrics.average_response_time = random.uniform(200, 800)
        else:
            metrics.task_queue_size = random.randint(0, 2)
            metrics.tasks_processed_per_minute = random.uniform(5, 15)
            metrics.average_response_time = random.uniform(100, 300)
        
        # Error rate simulation
        metrics.error_rate = max(0, random.uniform(0, 3) if workflow_active else random.uniform(0, 1))
        
        # Health score calculation
        cpu_penalty = max(0, (metrics.cpu_usage - 80) * 0.5)
        memory_penalty = max(0, (metrics.memory_usage - 70) * 0.3)
        error_penalty = metrics.error_rate * 5
        metrics.health_score = max(0, 100 - cpu_penalty - memory_penalty - error_penalty)
        
        # Update historical data
        metrics.cpu_history.append(metrics.cpu_usage)
        metrics.memory_history.append(metrics.memory_usage)
        metrics.response_time_history.append(metrics.average_response_time)
        
        metrics.last_updated = datetime.now()
    
    def log_communication(self, from_agent: str, to_agent: str, message_type: str):
        """Log agent communication with realistic network metrics"""
        communication = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            data_size=random.randint(1024, 8192),  # 1-8KB
            latency_ms=random.uniform(5, 50),
            timestamp=datetime.now(),
            status="success" if random.random() > 0.05 else "failed"  # 95% success rate
        )
        self.communication_log.append(communication)
    
    def log_decision(self, agent_id: str, decision_type: str, reasoning: str):
        """Log AI decision making process"""
        decision = WorkflowDecision(
            agent_id=agent_id,
            decision_type=decision_type,
            confidence_score=random.uniform(0.7, 0.98),
            reasoning=reasoning,
            alternatives_considered=[f"alternative_{i}" for i in range(random.randint(2, 4))],
            timestamp=datetime.now()
        )
        self.decision_log.append(decision)
    
    def predict_bottlenecks(self):
        """Predict potential agent bottlenecks using simple analytics"""
        predictions = {}
        
        for agent_id, metrics in self.agent_metrics.items():
            risk_score = 0
            
            # CPU trend analysis
            if len(metrics.cpu_history) >= 10:
                recent_cpu = list(metrics.cpu_history)[-10:]
                cpu_trend = (recent_cpu[-1] - recent_cpu[0]) / 10
                if cpu_trend > 2:  # Increasing CPU usage
                    risk_score += 30
            
            # Memory pressure
            if metrics.memory_usage > 70:
                risk_score += 25
            
            # Queue buildup
            if metrics.task_queue_size > 5:
                risk_score += 20
            
            # Response time degradation
            if metrics.average_response_time > 500:
                risk_score += 15
            
            predictions[agent_id] = {
                'risk_score': min(100, risk_score),
                'predicted_issues': self._get_predicted_issues(risk_score),
                'recommendations': self._get_recommendations(agent_id, risk_score)
            }
        
        return predictions
    
    def _get_predicted_issues(self, risk_score: int) -> List[str]:
        """Get predicted issues based on risk score"""
        issues = []
        if risk_score > 70:
            issues.extend(['Performance Degradation', 'Potential Timeout'])
        if risk_score > 50:
            issues.extend(['Queue Buildup', 'Resource Pressure'])
        if risk_score > 30:
            issues.append('Elevated Response Time')
        return issues
    
    def _get_recommendations(self, agent_id: str, risk_score: int) -> List[str]:
        """Get recommendations based on agent and risk"""
        recommendations = []
        if risk_score > 70:
            recommendations.extend([
                f'Scale {agent_id} resources',
                'Enable load balancing',
                'Activate backup agent'
            ])
        if risk_score > 50:
            recommendations.extend([
                'Monitor closely',
                'Prepare for scaling'
            ])
        return recommendations
    
    def get_network_topology_data(self):
        """Get comprehensive network topology data for visualization"""
        # Update all agent metrics
        active_workflows = random.choice([True, False])  # Simulate workflow activity
        for agent_id in self.agent_metrics.keys():
            self.update_agent_metrics(agent_id, active_workflows)
        
        # Get recent communications
        recent_comms = [c for c in self.communication_log if 
                       (datetime.now() - c.timestamp).seconds < 300]  # Last 5 minutes
        
        # Calculate network statistics
        network_stats = {
            'total_messages': len(recent_comms),
            'success_rate': len([c for c in recent_comms if c.status == 'success']) / max(1, len(recent_comms)) * 100,
            'average_latency': sum(c.latency_ms for c in recent_comms) / max(1, len(recent_comms)),
            'data_throughput': sum(c.data_size for c in recent_comms)  # bytes in last 5 min
        }
        
        # Convert deque to list for JSON serialization
        agents_data = {}
        for agent_id, metrics in self.agent_metrics.items():
            agent_dict = asdict(metrics)
            agent_dict['cpu_history'] = list(metrics.cpu_history)
            agent_dict['memory_history'] = list(metrics.memory_history)
            agent_dict['response_time_history'] = list(metrics.response_time_history)
            agent_dict['last_updated'] = metrics.last_updated.isoformat()
            agents_data[agent_id] = agent_dict
        
        return {
            'agents': agents_data,
            'communications': [asdict(c) for c in list(self.communication_log)[-20:]],  # Last 20
            'decisions': [asdict(d) for d in list(self.decision_log)[-10:]],  # Last 10
            'network_stats': network_stats,
            'bottleneck_predictions': self.predict_bottlenecks(),
            'topology_health': self._calculate_topology_health()
        }
    
    def _calculate_topology_health(self) -> Dict[str, Any]:
        """Calculate overall topology health metrics"""
        agent_healths = [m.health_score for m in self.agent_metrics.values()]
        
        return {
            'overall_health': sum(agent_healths) / len(agent_healths),
            'critical_agents': [aid for aid, m in self.agent_metrics.items() if m.health_score < 70],
            'healthy_agents': [aid for aid, m in self.agent_metrics.items() if m.health_score >= 85],
            'total_task_queue': sum(m.task_queue_size for m in self.agent_metrics.values()),
            'total_throughput': sum(m.tasks_processed_per_minute for m in self.agent_metrics.values())
        }

class EnhancedTimelineManager:
    """Manages timeline calculation and progress tracking"""
    
    def __init__(self):
        self.data_start_time: Optional[datetime] = None
        self.data_end_time: Optional[datetime] = None
        self.processing_start_time: Optional[datetime] = None
        self.total_data_timespan_seconds: float = 0.0
        self.processed_time_seconds: float = 0.0
        
    def initialize_timeline(self, anomalies_data: Dict[str, List[Dict]]):
        """Initialize timeline from anomaly data"""
        try:
            all_timestamps = []
            
            for component in ['amf', 'smf', 'upf']:
                component_anomalies = anomalies_data.get(component, [])
                for anomaly_entry in component_anomalies:
                    if anomaly_entry.get('anomalies'):
                        timestamp = pd.to_datetime(anomaly_entry['timestamp'])
                        all_timestamps.append(timestamp)
            
            if all_timestamps:
                self.data_start_time = min(all_timestamps)
                self.data_end_time = max(all_timestamps)
                self.total_data_timespan_seconds = (self.data_end_time - self.data_start_time).total_seconds()
                self.processing_start_time = datetime.now()
                
                logger.info(f"📅 Timeline initialized:")
                logger.info(f"   Data range: {self.data_start_time} to {self.data_end_time}")
                logger.info(f"   Total timespan: {self.total_data_timespan_seconds:.1f} seconds")
                
        except Exception as e:
            logger.error(f"Error initializing timeline: {str(e)}")
    
    def calculate_timeline_progress(self, current_position_timestamp: datetime) -> Dict[str, Any]:
        """Calculate progress based on data timeline position"""
        try:
            if not self.data_start_time or not self.data_end_time:
                return self._get_default_progress()
            
            # Calculate how far through the data timeline we've progressed
            if current_position_timestamp <= self.data_start_time:
                timeline_progress = 0.0
            elif current_position_timestamp >= self.data_end_time:
                timeline_progress = 100.0
            else:
                processed_span = (current_position_timestamp - self.data_start_time).total_seconds()
                timeline_progress = (processed_span / self.total_data_timespan_seconds) * 100.0
            
            # Calculate processing time
            processing_elapsed = (datetime.now() - self.processing_start_time).total_seconds() if self.processing_start_time else 0
            
            # Estimate remaining time based on processing rate
            if timeline_progress > 0 and timeline_progress < 100:
                estimated_total_time = processing_elapsed / (timeline_progress / 100.0)
                eta_seconds = estimated_total_time - processing_elapsed
            else:
                eta_seconds = 0
            
            return {
                'timeline_progress_percentage': min(100.0, max(0.0, timeline_progress)),
                'data_start_time': self.data_start_time.isoformat() if self.data_start_time else None,
                'data_end_time': self.data_end_time.isoformat() if self.data_end_time else None,
                'current_position_timestamp': current_position_timestamp.isoformat(),
                'processing_start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
                'processing_elapsed_seconds': processing_elapsed,
                'eta_seconds': max(0, eta_seconds),
                'total_data_timespan_seconds': self.total_data_timespan_seconds
            }
            
        except Exception as e:
            logger.error(f"Error calculating timeline progress: {str(e)}")
            return self._get_default_progress()
    
    def _get_default_progress(self) -> Dict[str, Any]:
        """Return default progress when timeline data unavailable"""
        return {
            'timeline_progress_percentage': 0.0,
            'data_start_time': None,
            'data_end_time': None,
            'current_position_timestamp': datetime.now().isoformat(),
            'processing_start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
            'processing_elapsed_seconds': 0.0,
            'eta_seconds': 0.0,
            'total_data_timespan_seconds': 0.0
        }

class EnhancedWorkflowManager:
    """Manages workflow execution and tracking"""
    
    def __init__(self, max_history: int = 1000):
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.completed_workflows: List[WorkflowExecution] = []
        self.max_history = max_history
        self.workflow_counter = 0
        
        # Agent to workflow status mapping
        self.agent_to_status = {
            'diagnostic': WorkflowStatus.DIAGNOSTIC,
            'planning': WorkflowStatus.PLANNING,
            'execution': WorkflowStatus.EXECUTING,
            'validation': WorkflowStatus.VALIDATING
        }
        
    def create_workflow(self, anomaly_event: AnomalyEvent) -> WorkflowExecution:
        """Create new workflow for anomaly event"""
        try:
            self.workflow_counter += 1
            workflow_id = f"wf_{anomaly_event.component.lower()}_{self.workflow_counter:04d}_{datetime.now().strftime('%H%M%S')}"
            
            # Create workflow steps
            steps = [
                WorkflowStep(
                    step_id=f"{workflow_id}_diagnostic",
                    agent="diagnostic",
                    description=f"Analyze {anomaly_event.component} anomaly patterns"
                ),
                WorkflowStep(
                    step_id=f"{workflow_id}_planning", 
                    agent="planning",
                    description=f"Generate remediation plan for {anomaly_event.component}"
                ),
                WorkflowStep(
                    step_id=f"{workflow_id}_execution",
                    agent="execution", 
                    description=f"Execute automated fix for {anomaly_event.component}"
                ),
                WorkflowStep(
                    step_id=f"{workflow_id}_validation",
                    agent="validation",
                    description=f"Validate {anomaly_event.component} recovery status"
                )
            ]
            
            workflow = WorkflowExecution(
                workflow_id=workflow_id,
                anomaly_event=anomaly_event,
                steps=steps
            )
            
            self.active_workflows[workflow_id] = workflow
            
            logger.info(f"🔄 Created workflow: {workflow_id}")
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            raise
    
    def advance_workflow_step(self, workflow_id: str, agent: str, output: str = None, error: str = None) -> bool:
        """Advance workflow to next step"""
        try:
            if workflow_id not in self.active_workflows:
                return False
                
            workflow = self.active_workflows[workflow_id]
            current_step = workflow.steps[workflow.current_step_index]
            
            if current_step.agent == agent:
                # Complete current step
                current_step.completed_at = datetime.now()
                current_step.duration_seconds = (current_step.completed_at - current_step.started_at).total_seconds() if current_step.started_at else 0
                current_step.output = output
                current_step.error = error
                current_step.status = WorkflowStepStatus.FAILED if error else WorkflowStepStatus.COMPLETED
                
                # Update workflow status
                if error:
                    workflow.status = WorkflowStatus.FAILED
                    workflow.completed_at = datetime.now()
                    self._complete_workflow(workflow_id)
                else:
                    # Move to next step
                    workflow.current_step_index += 1
                    if workflow.current_step_index >= len(workflow.steps):
                        # Workflow completed
                        workflow.status = WorkflowStatus.COMPLETED
                        workflow.completed_at = datetime.now()
                        workflow.total_duration_seconds = (workflow.completed_at - workflow.started_at).total_seconds() if workflow.started_at else 0
                        self._complete_workflow(workflow_id)
                    else:
                        # Start next step
                        next_step = workflow.steps[workflow.current_step_index]
                        next_step.status = WorkflowStepStatus.ACTIVE
                        next_step.started_at = datetime.now()
                        
                        # Use the mapping to get the correct WorkflowStatus
                        workflow.status = self.agent_to_status.get(next_step.agent, WorkflowStatus.EXECUTING)
                
                return True
                
        except Exception as e:
            logger.error(f"Error advancing workflow step: {str(e)}")
            return False
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution"""
        try:
            if workflow_id not in self.active_workflows:
                return False
                
            workflow = self.active_workflows[workflow_id]
            workflow.started_at = datetime.now()
            workflow.status = WorkflowStatus.DIAGNOSTIC
            
            # Start first step
            first_step = workflow.steps[0]
            first_step.status = WorkflowStepStatus.ACTIVE
            first_step.started_at = datetime.now()
            
            logger.info(f"▶️  Started workflow: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting workflow: {str(e)}")
            return False
    
    def _complete_workflow(self, workflow_id: str):
        """Move workflow from active to completed"""
        try:
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows.pop(workflow_id)
                self.completed_workflows.append(workflow)
                
                # Maintain history limit
                if len(self.completed_workflows) > self.max_history:
                    self.completed_workflows = self.completed_workflows[-self.max_history:]
                
                logger.info(f"✅ Completed workflow: {workflow_id} - Status: {workflow.status.value}")
                
        except Exception as e:
            logger.error(f"Error completing workflow: {str(e)}")
    
    def get_workflow_summary(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """Get paginated workflow summary"""
        try:
            # Combine active and completed workflows
            all_workflows = list(self.active_workflows.values()) + self.completed_workflows
            all_workflows.sort(key=lambda w: w.created_at, reverse=True)
            
            # Pagination
            total_workflows = len(all_workflows)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_workflows = all_workflows[start_idx:end_idx]
            
            # Convert to serializable format
            workflow_data = []
            for workflow in page_workflows:
                workflow_dict = {
                    'workflow_id': workflow.workflow_id,
                    'status': workflow.status.value,
                    'created_at': workflow.created_at.isoformat(),
                    'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                    'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                    'total_duration_seconds': workflow.total_duration_seconds,
                    'anomaly': {
                        'event_id': workflow.anomaly_event.event_id,
                        'component': workflow.anomaly_event.component,
                        'severity': workflow.anomaly_event.severity,
                        'timestamp': workflow.anomaly_event.timestamp.isoformat(),
                        'trigger_type': workflow.anomaly_event.trigger_type.value
                    },
                    'selected_playbook': workflow.selected_playbook,
                    'playbook_description': workflow.playbook_description,  # NEW: Enhanced info
                    'current_step_index': workflow.current_step_index,
                    'steps': [
                        {
                            'step_id': step.step_id,
                            'agent': step.agent,
                            'description': step.description,
                            'status': step.status.value,
                            'started_at': step.started_at.isoformat() if step.started_at else None,
                            'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                            'duration_seconds': step.duration_seconds,
                            'output': step.output,
                            'error': step.error
                        }
                        for step in workflow.steps
                    ]
                }
                workflow_data.append(workflow_dict)
            
            return {
                'workflows': workflow_data,
                'pagination': {
                    'current_page': page,
                    'page_size': page_size,
                    'total_workflows': total_workflows,
                    'total_pages': math.ceil(total_workflows / page_size) if page_size > 0 else 0,
                    'has_next': end_idx < total_workflows,
                    'has_previous': page > 1
                },
                'stats': {
                    'active_workflows': len(self.active_workflows),
                    'completed_workflows': len(self.completed_workflows),
                    'total_workflows': total_workflows
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting workflow summary: {str(e)}")
            return {
                'workflows': [],
                'pagination': {'current_page': 1, 'page_size': page_size, 'total_workflows': 0, 'total_pages': 0, 'has_next': False, 'has_previous': False},
                'stats': {'active_workflows': 0, 'completed_workflows': 0, 'total_workflows': 0}
            }

class NOCDataMonitor:
    """Enhanced NOC data monitor with timeline-based processing"""
    
    def __init__(self, data_path: str = "processed_data", poll_interval: float = 5.0):
        self.data_path = Path(data_path)
        self.poll_interval = poll_interval
        self.processed_anomalies = set()
        self.event_callbacks = []
        self.running = False
        self._monitor_task = None
        self._timeline_task = None
        
        # FIXED: Track DISCOVERED anomaly counts that start from 0 and increment
        self.total_anomaly_counts = {'AMF': 0, 'SMF': 0, 'UPF': 0}
        self.discovered_counts = {'AMF': 0, 'SMF': 0, 'UPF': 0}  # NEW: Discovered over time
        
        # Enhanced timeline tracking
        self.timeline_manager = EnhancedTimelineManager()
        self.current_timeline_position = None
        self.min_time = None
        self.max_time = None
        
        # Time tracking
        self.start_time = datetime.now()
        self.processing_start_time = None
        
        # Store original anomalies data for timeline calculation
        self.original_anomalies_data = {}
        
        # Speed control and timeline progression
        self.processing_speed = 1.0  # 1x, 5x, 10x, 100x
        self.timeline_interval = 2.0  # Base seconds between timeline updates
        
        # Recent anomalies for tracking
        self.recent_anomalies = deque(maxlen=50)
        
    def add_event_callback(self, callback):
        """Add callback function to be called when new anomalies are detected"""
        self.event_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Enhanced start monitoring with timeline-based processing"""
        self.running = True
        self.processing_start_time = datetime.now()
        
        try:
            # Do enhanced initial scan
            await self._enhanced_initial_scan()
            
            # Start monitoring loop
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            # Start timeline processing (replaces autonomous processing)
            self._timeline_task = asyncio.create_task(self._timeline_processing_loop())
            
            logger.info(f"Started enhanced NOC monitoring on {self.data_path}")
            logger.info(f"Timeline anomaly counts: {self.total_anomaly_counts}")
            logger.info("📊 Timeline-based processing enabled")
            
        except Exception as e:
            logger.error(f"Error starting enhanced monitoring: {str(e)}")
            self.running = False
    
    async def stop_monitoring(self):
        """Enhanced stop monitoring"""
        self.running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
        if self._timeline_task:
            self._timeline_task.cancel()
            try:
                await self._timeline_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Stopped enhanced NOC data monitoring")
    
    async def _enhanced_initial_scan(self):
        """Enhanced initial scan with timeline initialization"""
        try:
            anomalies_file = self.data_path / "anomalies.pkl"
            if not anomalies_file.exists():
                logger.warning(f"Anomalies file not found: {anomalies_file}")
                return
            
            with open(anomalies_file, 'rb') as f:
                anomalies = pickle.load(f)
            
            # Store original data for timeline calculation
            self.original_anomalies_data = anomalies
            
            # Find min/max timestamps
            all_timestamps = []
            for component in ['amf', 'smf', 'upf']:
                component_anomalies = [entry for entry in anomalies.get(component, []) 
                                     if entry.get('anomalies')]
                self.total_anomaly_counts[component.upper()] = len(component_anomalies)
                
                for anomaly_entry in component_anomalies:
                    timestamp = pd.to_datetime(anomaly_entry['timestamp'])
                    all_timestamps.append(timestamp)
            
            if all_timestamps:
                self.min_time = min(all_timestamps)
                self.max_time = max(all_timestamps)
                self.current_timeline_position = self.min_time
                
                # Initialize timeline manager
                self.timeline_manager.initialize_timeline(anomalies)
                
                logger.info(f"Timeline initialized:")
                logger.info(f"   Time range: {self.min_time} to {self.max_time}")
                logger.info(f"   Total anomalies: {sum(self.total_anomaly_counts.values())}")
                logger.info(f"   Duration: {(self.max_time - self.min_time).total_seconds():.1f} seconds")
            else:
                logger.warning("No anomalies found for timeline initialization")
                
        except Exception as e:
            logger.error(f"Error in enhanced initial scan: {str(e)}")
            
    def set_processing_speed(self, speed: float):
        """Set processing speed multiplier"""
        self.processing_speed = speed
        logger.info(f"🚀 Processing speed set to {speed}x")
    
    async def _timeline_processing_loop(self):
        """Timeline progression loop"""
        await asyncio.sleep(3)  # Wait 3 seconds before starting
        
        while self.running and self.min_time and self.max_time:
            try:
                # Calculate sleep time based on speed
                sleep_time = max(0.1, self.timeline_interval / self.processing_speed)
                await asyncio.sleep(sleep_time)
                
                # Advance timeline position
                if self.current_timeline_position < self.max_time:
                    # Advance by 1 minute in data time per iteration (adjustable)
                    time_increment = timedelta(minutes=1)
                    self.current_timeline_position += time_increment
                    
                    # Process any anomalies that occurred at this time
                    await self._process_anomalies_at_current_time()
                else:
                    # Reached end, restart from beginning
                    self.current_timeline_position = self.min_time
                    self.discovered_counts = {'AMF': 0, 'SMF': 0, 'UPF': 0}  # RESET discovered counts
                    self.processed_anomalies = set()
                    logger.info("🔄 Timeline restarted from beginning")
                
            except Exception as e:
                logger.error(f"Error in timeline processing loop: {str(e)}")
    
    async def _process_anomalies_at_current_time(self):
        """Process anomalies that occurred at or before current timeline position"""
        try:
            if not self.original_anomalies_data:
                return
                
            # Process anomalies - accumulate up to current time
            for component in ['amf', 'smf', 'upf']:
                component_anomalies = [
                    entry for entry in self.original_anomalies_data.get(component, [])
                    if entry.get('anomalies') and 
                    pd.to_datetime(entry['timestamp']) <= self.current_timeline_position
                ]
                
                # Update discovered count to match current timeline position
                new_count = len(component_anomalies)
                if new_count > self.discovered_counts[component.upper()]:
                    # We have new anomalies to process
                    newly_discovered = component_anomalies[self.discovered_counts[component.upper()]:]
                    
                    for anomaly_entry in newly_discovered:
                        timestamp = pd.to_datetime(anomaly_entry['timestamp'])
                        anomaly_id = f"{component}_{timestamp.isoformat()}_{hash(str(anomaly_entry['anomalies']))}"
                        
                        if anomaly_id not in self.processed_anomalies:
                            self.processed_anomalies.add(anomaly_id)
                            
                            # Create and trigger anomaly event
                            await self._create_timeline_anomaly_event(component, anomaly_entry, anomaly_id)
                    
                    # Update count
                    self.discovered_counts[component.upper()] = new_count
                    
        except Exception as e:
            logger.error(f"Error processing anomalies at current time: {str(e)}")
    
    async def _create_timeline_anomaly_event(self, component: str, anomaly_entry: Dict, anomaly_id: str):
        """Create and process a timeline-based anomaly event"""
        try:
            timestamp = pd.to_datetime(anomaly_entry['timestamp'])
            severity = self._calculate_severity(anomaly_entry['anomalies'])
            
            # Load RCA results for context
            rca_results = self._load_rca_results()
            closest_rca = self._find_closest_rca(timestamp, rca_results)
            
            # Create event for timeline anomaly
            event = AnomalyEvent(
                event_id=anomaly_id,
                trigger_type=AnomalyTriggerType.AUTONOMOUS_PROCESSING,
                component=component.upper(),
                timestamp=timestamp,
                anomaly_data=anomaly_entry['anomalies'],
                recommendations=closest_rca.get('recommendations') if closest_rca else f"Optimize {component.upper()} performance and resource utilization",
                rca_result=closest_rca.get('rca') if closest_rca else f"Performance degradation detected in {component.upper()} component",
                severity=severity
            )
            
            # Track recent anomaly
            self.recent_anomalies.append({
                'component': component.upper(),
                'timestamp': timestamp,
                'severity': severity
            })
            
            logger.info(f"📊 TIMELINE PROCESSING: {component.upper()} - {severity} at {timestamp}")
            
            # Trigger the full agent workflow
            await self._trigger_event_callbacks(event)
            
        except Exception as e:
            logger.error(f"Error creating timeline anomaly event: {str(e)}")
    
    def _load_rca_results(self) -> Dict:
        """Load RCA results from file"""
        try:
            rca_file = self.data_path / "rca_results.pkl"
            if rca_file.exists():
                with open(rca_file, 'rb') as f:
                    rca_data = pickle.load(f)
                    return {pd.to_datetime(result['timestamp']): result for result in rca_data}
            return {}
        except Exception as e:
            logger.error(f"Error loading RCA results: {str(e)}")
            return {}
    
    def _find_closest_rca(self, timestamp: datetime, rca_results: Dict) -> Optional[Dict]:
        """Find the closest RCA result to a given timestamp"""
        closest_rca = None
        min_time_diff = timedelta(minutes=10)
        
        for rca_time, rca_data in rca_results.items():
            time_diff = abs(rca_time - timestamp)
            if time_diff < min_time_diff:
                closest_rca = rca_data
                min_time_diff = time_diff
                
        return closest_rca
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_for_new_anomalies()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.poll_interval)
    
    async def _check_for_new_anomalies(self):
        """Check for new anomalies in the processed data"""
        # This method keeps the original functionality for live anomaly detection
        try:
            anomalies_file = self.data_path / "anomalies.pkl"
            if not anomalies_file.exists():
                return
            
            with open(anomalies_file, 'rb') as f:
                current_anomalies = pickle.load(f)
            
            rca_results = self._load_rca_results()
            
            # Check each component for new anomalies
            for component in ['amf', 'smf', 'upf']:
                await self._process_component_anomalies(
                    component, 
                    current_anomalies.get(component, []),
                    rca_results
                )
                
        except Exception as e:
            logger.error(f"Error checking anomalies: {str(e)}")
    
    async def _process_component_anomalies(self, component: str, anomalies: List[Dict], rca_results: Dict):
        """Process anomalies for a specific component (original functionality)"""
        for anomaly_entry in anomalies:
            if not anomaly_entry.get('anomalies'):
                continue
                
            try:
                timestamp = pd.to_datetime(anomaly_entry['timestamp'])
                anomaly_id = f"{component}_{timestamp.isoformat()}_{hash(str(anomaly_entry['anomalies']))}"
                
                # Skip if already processed
                if anomaly_id in self.processed_anomalies:
                    continue
                    
                # This would be for truly new anomalies that appear in the data
                # For now, we'll skip since we're focusing on autonomous processing
                pass
                
            except Exception as e:
                logger.error(f"Error processing anomaly in {component}: {str(e)}")
                continue
    
    def _calculate_severity(self, anomalies: Dict[str, Any]) -> str:
        """Calculate severity based on anomaly characteristics"""
        try:
            max_deviation = 0
            for metric, anomaly_info in anomalies.items():
                if isinstance(anomaly_info, dict) and 'deviation_sigma' in anomaly_info:
                    deviation = abs(anomaly_info['deviation_sigma'])
                    max_deviation = max(max_deviation, deviation)
            
            if max_deviation > 4.0:
                return "CRITICAL"
            elif max_deviation > 3.0:
                return "HIGH"
            elif max_deviation > 2.0:
                return "MEDIUM"
            else:
                return "LOW"
        except Exception as e:
            logger.error(f"Error calculating severity: {str(e)}")
            return "MEDIUM"
    
    async def _trigger_event_callbacks(self, event: AnomalyEvent):
        """Trigger all registered event callbacks"""
        logger.info(f"🚨 ANOMALY EVENT: {event.component} - {event.severity}")
        logger.info(f"   Event ID: {event.event_id}")
        logger.info(f"   Trigger: {event.trigger_type.value}")
        
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {str(e)}")

    async def trigger_manual_event(self, component: str, severity: str = "HIGH"):
        """Manually trigger an anomaly event for testing"""
        try:
            event_id = f"manual_{component.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            event = AnomalyEvent(
                event_id=event_id,
                trigger_type=AnomalyTriggerType.MANUAL_TRIGGER,
                component=component.upper(),
                timestamp=datetime.now(),
                anomaly_data={
                    "manual_trigger": {
                        "value": 100,
                        "mean": 50,
                        "std": 10,
                        "deviation_sigma": 5.0,
                        "detection_method": "manual"
                    }
                },
                recommendations=f"Manual test triggered for {component} component",
                rca_result=f"Manual testing of {component} remediation workflow",
                severity=severity
            )
            
            await self._trigger_event_callbacks(event)
            return event
        except Exception as e:
            logger.error(f"Error triggering manual event: {str(e)}")
            raise

    def get_processing_progress(self) -> Dict[str, Any]:
        """Get current processing progress with timeline data"""
        try:
            # FIXED: Use discovered counts instead of processed counts
            total_anomalies = sum(self.total_anomaly_counts.values())
            discovered_anomalies = sum(self.discovered_counts.values())
            progress_percentage = (discovered_anomalies / total_anomalies * 100) if total_anomalies > 0 else 0
            
            # Get timeline progress
            timeline_data = {}
            if self.current_timeline_position and self.min_time and self.max_time:
                timeline_data = self.timeline_manager.calculate_timeline_progress(self.current_timeline_position)
            elif self.min_time:  # Fallback to start time
                timeline_data = self.timeline_manager.calculate_timeline_progress(self.min_time)
            
            return {
                'total_anomalies': total_anomalies,
                'discovered_anomalies': discovered_anomalies,  # CHANGED from processed_anomalies
                'remaining_anomalies': total_anomalies - discovered_anomalies,
                'progress_percentage': round(progress_percentage, 1),
                'processing_start_time': self.processing_start_time.isoformat() if self.processing_start_time else None,
                'current_time': datetime.now().isoformat(),
                'current_timeline_position': self.current_timeline_position.isoformat() if self.current_timeline_position else None,
                'timeline_data': timeline_data,
                'processing_speed': self.processing_speed,
                'min_time': self.min_time.isoformat() if self.min_time else None,
                'max_time': self.max_time.isoformat() if self.max_time else None,
                'by_component': {
                    'AMF': {
                        'total': self.total_anomaly_counts['AMF'],
                        'discovered': self.discovered_counts['AMF'],  # CHANGED from processed
                        'remaining': self.total_anomaly_counts['AMF'] - self.discovered_counts['AMF']
                    },
                    'SMF': {
                        'total': self.total_anomaly_counts['SMF'],
                        'discovered': self.discovered_counts['SMF'],
                        'remaining': self.total_anomaly_counts['SMF'] - self.discovered_counts['SMF']
                    },
                    'UPF': {
                        'total': self.total_anomaly_counts['UPF'],
                        'discovered': self.discovered_counts['UPF'],
                        'remaining': self.total_anomaly_counts['UPF'] - self.discovered_counts['UPF']
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting processing progress: {str(e)}")
            return {
                'total_anomalies': 0,
                'discovered_anomalies': 0,
                'remaining_anomalies': 0,
                'progress_percentage': 0,
                'processing_start_time': None,
                'current_time': datetime.now().isoformat(),
                'current_timeline_position': None,
                'timeline_data': {},
                'processing_speed': 1.0,
                'min_time': None,
                'max_time': None,
                'by_component': {'AMF': {'total': 0, 'discovered': 0, 'remaining': 0},
                               'SMF': {'total': 0, 'discovered': 0, 'remaining': 0},
                               'UPF': {'total': 0, 'discovered': 0, 'remaining': 0}}
            }
    
    def get_recent_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent anomalies for display"""
        return list(self.recent_anomalies)[-limit:]

class AnsiblePlaybookExecutor:
    """Executes Ansible playbooks based on recommendations"""
    
    def __init__(self, playbook_dir: str = "playbooks"):
        self.playbook_dir = Path(playbook_dir)
        self.ansible_available = self._check_ansible_availability()
        
        # ENHANCED: Playbook descriptions for better UI display
        self.playbook_descriptions = {
            'scale_amf_resources.yml': 'Scale AMF CPU/Memory Resources',
            'restart_amf_service.yml': 'Restart AMF Service Components',
            'restart_smf_service.yml': 'Restart SMF Service Components',
            'adjust_upf_load_balancing.yml': 'Adjust UPF Load Balancing',
            'resource_optimization.yml': 'General Resource Optimization',
            'test_playbooks.sh': 'Test Playbook Execution'
        }
        
    def _check_ansible_availability(self) -> bool:
        """Check if Ansible is available"""
        try:
            return shutil.which('ansible-playbook') is not None
        except Exception:
            return False
    
    def get_playbook_description(self, playbook_name: str) -> str:
        """Get human-readable description for playbook"""
        return self.playbook_descriptions.get(playbook_name, f"Execute {playbook_name}")
    
    async def execute_playbook(self, playbook_name: str, extra_vars: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an Ansible playbook"""
        try:
            playbook_path = self.playbook_dir / playbook_name
            
            if not playbook_path.exists():
                raise FileNotFoundError(f"Playbook not found: {playbook_name}")
            
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"📋 EXECUTING PLAYBOOK: {playbook_name}")
            logger.info(f"   Execution ID: {execution_id}")
            logger.info(f"   Extra vars: {extra_vars}")
            
            start_time = datetime.now()
            
            # Simulate execution
            await asyncio.sleep(3)  # Simulate execution time
            
            execution_result = {
                "execution_id": execution_id,
                "playbook": playbook_name,
                "playbook_description": self.get_playbook_description(playbook_name),  # NEW
                "status": "success",
                "started_at": start_time.isoformat(),
                "duration_seconds": 3,
                "extra_vars": extra_vars or {},
                "output": f"AUTONOMOUS EXECUTION: Successfully executed {playbook_name}",
                "changed": True,
                "failed": False,
                "mode": "autonomous"
            }
            
            logger.info(f"✅ PLAYBOOK EXECUTION SUCCESS")
            logger.info(f"   Duration: {execution_result['duration_seconds']:.1f} seconds")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing playbook {playbook_name}: {str(e)}")
            return {
                "execution_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "playbook": playbook_name,
                "playbook_description": self.get_playbook_description(playbook_name),
                "status": "error",
                "started_at": datetime.now().isoformat(),
                "duration_seconds": 0,
                "extra_vars": extra_vars or {},
                "output": f"Error: {str(e)}",
                "error": str(e),
                "mode": "error"
            }

class EnhancedNOCDashboard:
    """Enhanced NOC Dashboard with advanced agent topology"""
    
    def __init__(self):
        self.noc_monitor = None
        self.ansible_executor = None
        self.execution_history = []
        self.agent_interactions = []
        
        # Enhanced workflow management
        self.workflow_manager = EnhancedWorkflowManager()
        
        # Enhanced agent topology manager
        self.topology_manager = EnhancedAgentTopologyManager()
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        self.agent_states = {
            'diagnostic': {'status': 'active', 'current_task': 'Monitoring telemetry', 'tasks_completed': 0},
            'planning': {'status': 'idle', 'current_task': 'Ready', 'tasks_completed': 0},
            'execution': {'status': 'idle', 'current_task': 'Ready', 'tasks_completed': 0},
            'validation': {'status': 'idle', 'current_task': 'Ready', 'tasks_completed': 0}
        }
        self.stats = {
            'anomalies_processed': 0,
            'playbooks_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'start_time': None,
            'agents_active': 4,
            'workflows_running': 0,
            'avg_resolution_time': 45.2
        }
        self.app = web.Application()
        self._setup_routes()
        
        # Load dashboard HTML from file
        self.dashboard_html = self._load_dashboard_html()
    
    def _load_dashboard_html(self) -> str:
        """Load dashboard HTML from external file"""
        try:
            # Look for dashboard.html in the same directory as the script
            html_path = Path(__file__).parent / "dashboard.html"
            if html_path.exists():
                with open(html_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"Dashboard HTML not found at {html_path}, using fallback")
                return self._get_fallback_html()
        except Exception as e:
            logger.error(f"Error loading dashboard HTML: {str(e)}")
            return self._get_fallback_html()
    
    def _get_fallback_html(self) -> str:
        """Get fallback HTML if external file not found"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Autonomous 5G NOC</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #1a1a1a;
                    color: #ffffff;
                    padding: 20px;
                    text-align: center;
                }
                h1 { color: #3b82f6; }
                .error {
                    background: #ef4444;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px auto;
                    max-width: 600px;
                }
            </style>
        </head>
        <body>
            <h1>Enhanced Autonomous 5G NOC</h1>
            <div class="error">
                <h2>Dashboard HTML Not Found</h2>
                <p>Please ensure dashboard.html is in the same directory as the main script.</p>
                <p>The dashboard is still operational via API endpoints.</p>
            </div>
        </body>
        </html>
        """
    
    def _setup_routes(self):
        """Setup web routes with enhanced workflow endpoints"""
        self.app.router.add_get('/', self.dashboard_handler)
        self.app.router.add_get('/api/status', self.status_handler)
        self.app.router.add_get('/api/stats', self.stats_handler)
        self.app.router.add_get('/api/executions', self.executions_handler)
        self.app.router.add_get('/api/progress', self.progress_handler)
        self.app.router.add_get('/api/agents', self.agents_handler)
        self.app.router.add_get('/api/agent-interactions', self.agent_interactions_handler)
        
        # Enhanced workflow endpoints
        self.app.router.add_get('/api/workflows', self.workflows_handler)
        self.app.router.add_get('/api/workflows/{workflow_id}', self.workflow_detail_handler)
        
        # NEW: Enhanced agent topology endpoint
        self.app.router.add_get('/api/enhanced-agent-topology', self.enhanced_agent_topology_handler)
        
        # NEW: Real-time metrics endpoint
        self.app.router.add_get('/api/metrics', self.metrics_handler)
        
        # Speed control endpoint
        self.app.router.add_post('/api/speed/{speed}', self.speed_handler)
        
        self.app.router.add_post('/api/trigger/{component}', self.trigger_handler)
        self.app.router.add_get('/health', self.health_handler)
        
        # Serve static files (dashboard.js)
        self.app.router.add_get('/dashboard.js', self.serve_dashboard_js)
        
        # Setup CORS
        try:
            cors = aiohttp_cors.setup(self.app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            
            for route in list(self.app.router.routes()):
                cors.add(route)
        except Exception as e:
            logger.warning(f"Could not setup CORS: {str(e)}")
    
    async def serve_dashboard_js(self, request):
        """Serve dashboard.js file"""
        try:
            js_path = Path(__file__).parent / "dashboard.js"
            if js_path.exists():
                with open(js_path, 'r', encoding='utf-8') as f:
                    js_content = f.read()
                return web_response.Response(
                    text=js_content,
                    content_type='application/javascript',
                    charset='utf-8'
                )
            else:
                return web_response.Response(
                    text="console.error('dashboard.js not found');",
                    content_type='application/javascript',
                    status=404
                )
        except Exception as e:
            logger.error(f"Error serving dashboard.js: {str(e)}")
            return web_response.Response(
                text=f"console.error('Error loading dashboard.js: {str(e)}');",
                content_type='application/javascript',
                status=500
            )
    
    async def enhanced_agent_topology_handler(self, request):
        """Enhanced agent topology data endpoint"""
        try:
            # Simulate some communications based on current workflows
            if len(self.workflow_manager.active_workflows) > 0:
                # Simulate agent communications for active workflows
                self.topology_manager.log_communication('diagnostic', 'planning', 'anomaly_report')
                self.topology_manager.log_communication('planning', 'execution', 'remediation_plan')
                self.topology_manager.log_decision('diagnostic', 'escalate', 'High severity anomaly detected')
            
            topology_data = self.topology_manager.get_network_topology_data()
            
            # Convert datetime objects to ISO strings for JSON serialization
            for comm in topology_data['communications']:
                if 'timestamp' in comm:
                    comm['timestamp'] = comm['timestamp'].isoformat() if hasattr(comm['timestamp'], 'isoformat') else str(comm['timestamp'])
            
            for decision in topology_data['decisions']:
                if 'timestamp' in decision:
                    decision['timestamp'] = decision['timestamp'].isoformat() if hasattr(decision['timestamp'], 'isoformat') else str(decision['timestamp'])
            
            return web.json_response(topology_data)
            
        except Exception as e:
            logger.error(f"Error in enhanced agent topology handler: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def metrics_handler(self, request):
        """Real-time metrics data endpoint"""
        try:
            # Generate simulated metrics based on current timeline
            if self.noc_monitor and self.noc_monitor.current_timeline_position:
                self.metrics_tracker.generate_simulated_metrics(self.noc_monitor.current_timeline_position)
            else:
                self.metrics_tracker.generate_simulated_metrics()
            
            # Add anomaly data from recent anomalies
            if self.noc_monitor:
                recent_anomalies = self.noc_monitor.get_recent_anomalies()
                for anomaly in recent_anomalies:
                    # Convert timestamp if needed
                    timestamp = anomaly.get('timestamp')
                    if hasattr(timestamp, 'isoformat'):
                        timestamp = timestamp.isoformat()
                    elif not isinstance(timestamp, str):
                        timestamp = str(timestamp)
                    
                    self.metrics_tracker.add_anomaly(
                        anomaly.get('component', 'unknown'),
                        timestamp,
                        anomaly.get('severity', 'MEDIUM')
                    )
            
            # Get metrics data
            metrics_data = self.metrics_tracker.get_metrics_data(window_size=30)
            
            return web.json_response(metrics_data)
            
        except Exception as e:
            logger.error(f"Error in metrics handler: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def dashboard_handler(self, request):
        """Serve the enhanced NOC dashboard"""
        return web_response.Response(text=self.dashboard_html, content_type='text/html')
    
    async def status_handler(self, request):
        """Return system status"""
        try:
            status = {
                "running": self.noc_monitor and self.noc_monitor.running,
                "ansible_available": self.ansible_executor.ansible_available if self.ansible_executor else False,
                "data_path": str(self.noc_monitor.data_path) if self.noc_monitor else None,
                "playbook_dir": str(self.ansible_executor.playbook_dir) if self.ansible_executor else None,
                "timestamp": datetime.now().isoformat(),
                "agents_active": 4,
                "autonomous_mode": True
            }
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error in status handler: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def stats_handler(self, request):
        """Return current statistics with discovered counts"""
        try:
            stats = dict(self.stats)
            
            # FIXED: Add discovered anomaly counts for UI
            if self.noc_monitor:
                stats['anomaly_counts'] = self.noc_monitor.total_anomaly_counts
                stats['discovered_anomaly_counts'] = self.noc_monitor.discovered_counts  # NEW
                
                # Fix timestamp serialization for recent_anomalies
                recent_anomalies = self.noc_monitor.get_recent_anomalies()
                stats['recent_anomalies'] = []
                for anomaly in recent_anomalies:
                    anomaly_copy = dict(anomaly)
                    # Convert pandas Timestamp to ISO string
                    if 'timestamp' in anomaly_copy:
                        if hasattr(anomaly_copy['timestamp'], 'isoformat'):
                            anomaly_copy['timestamp'] = anomaly_copy['timestamp'].isoformat()
                        else:
                            anomaly_copy['timestamp'] = str(anomaly_copy['timestamp'])
                    stats['recent_anomalies'].append(anomaly_copy)
            else:
                stats['anomaly_counts'] = {'AMF': 0, 'SMF': 0, 'UPF': 0}
                stats['discovered_anomaly_counts'] = {'AMF': 0, 'SMF': 0, 'UPF': 0}
                stats['recent_anomalies'] = []
            
            # Fix datetime serialization issue
            if stats['start_time']:
                start_time = stats['start_time']
                stats['runtime_seconds'] = (datetime.now() - start_time).total_seconds()
                stats['start_time'] = start_time.isoformat()
            else:
                stats['runtime_seconds'] = 0
                stats['start_time'] = None
            
            # Add dynamic stats
            stats['agents_active'] = 4
            stats['workflows_running'] = len([e for e in self.execution_history if 
                                           (datetime.now() - e['timestamp']).seconds < 300])
            stats['timestamp'] = datetime.now().isoformat()
            
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Error in stats handler: {str(e)}")
            return web.json_response({
                "error": str(e),
                "anomaly_counts": {'AMF': 0, 'SMF': 0, 'UPF': 0},
                "discovered_anomaly_counts": {'AMF': 0, 'SMF': 0, 'UPF': 0},
                "recent_anomalies": [],
                "anomalies_processed": 0,
                "playbooks_executed": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "runtime_seconds": 0,
                "start_time": None,
                "agents_active": 4,
                "workflows_running": 0,
                "timestamp": datetime.now().isoformat()
            }, status=500)
    
    async def executions_handler(self, request):
        """Return execution history"""
        try:
            executions = []
            for execution in self.execution_history:
                exec_copy = dict(execution)
                exec_copy['timestamp'] = execution['timestamp'].isoformat()
                
                if 'event' in exec_copy and exec_copy['event']:
                    event_copy = asdict(exec_copy['event'])
                    event_copy['timestamp'] = execution['event'].timestamp.isoformat()
                    event_copy['trigger_type'] = execution['event'].trigger_type.value
                    exec_copy['event'] = event_copy
                
                executions.append(exec_copy)
            
            return web.json_response(executions)
            
        except Exception as e:
            logger.error(f"Error in executions handler: {str(e)}")
            return web.json_response([], status=500)
    
    async def progress_handler(self, request):
        """Return processing progress"""
        try:
            if self.noc_monitor:
                progress = self.noc_monitor.get_processing_progress()
                return web.json_response(progress)
            else:
                return web.json_response({
                    'total_anomalies': 0,
                    'discovered_anomalies': 0,
                    'remaining_anomalies': 0,
                    'progress_percentage': 0,
                    'processing_start_time': None,
                    'current_time': datetime.now().isoformat(),
                    'latest_processed_timestamp': None,
                    'timeline_data': {},
                    'by_component': {'AMF': {'total': 0, 'discovered': 0, 'remaining': 0},
                                   'SMF': {'total': 0, 'discovered': 0, 'remaining': 0},
                                   'UPF': {'total': 0, 'discovered': 0, 'remaining': 0}}
                })
        except Exception as e:
            logger.error(f"Error in progress handler: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def agents_handler(self, request):
        """Return agent status information"""
        try:
            return web.json_response(self.agent_states)
        except Exception as e:
            logger.error(f"Error in agents handler: {str(e)}")
            return web.json_response({}, status=500)
    
    async def agent_interactions_handler(self, request):
        """Return agent interactions"""
        try:
            # Convert datetime objects to strings for JSON serialization
            interactions = []
            for interaction in self.agent_interactions:
                interaction_copy = asdict(interaction)
                interaction_copy['timestamp'] = interaction.timestamp.isoformat()
                interactions.append(interaction_copy)
            
            return web.json_response(interactions)
        except Exception as e:
            logger.error(f"Error in agent interactions handler: {str(e)}")
            return web.json_response([], status=500)
    
    async def workflows_handler(self, request):
        """Enhanced workflows endpoint with pagination"""
        try:
            # Get query parameters
            page = int(request.query.get('page', 1))
            page_size = int(request.query.get('page_size', 20))
            
            # Get workflow summary
            workflow_summary = self.workflow_manager.get_workflow_summary(page, page_size)
            
            return web.json_response(workflow_summary)
            
        except Exception as e:
            logger.error(f"Error in workflows handler: {str(e)}")
            return web.json_response({
                'workflows': [],
                'pagination': {'current_page': 1, 'page_size': 20, 'total_workflows': 0, 'total_pages': 0, 'has_next': False, 'has_previous': False},
                'stats': {'active_workflows': 0, 'completed_workflows': 0, 'total_workflows': 0}
            }, status=500)
    
    async def workflow_detail_handler(self, request):
        """Get detailed information about a specific workflow"""
        try:
            workflow_id = request.match_info['workflow_id']
            
            # Check active workflows first
            if workflow_id in self.workflow_manager.active_workflows:
                workflow = self.workflow_manager.active_workflows[workflow_id]
            else:
                # Check completed workflows
                workflow = None
                for w in self.workflow_manager.completed_workflows:
                    if w.workflow_id == workflow_id:
                        workflow = w
                        break
            
            if not workflow:
                return web.json_response({'error': 'Workflow not found'}, status=404)
            
            # Return detailed workflow information
            workflow_detail = {
                'workflow_id': workflow.workflow_id,
                'status': workflow.status.value,
                'created_at': workflow.created_at.isoformat(),
                'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                'total_duration_seconds': workflow.total_duration_seconds,
                'anomaly_event': {
                    'event_id': workflow.anomaly_event.event_id,
                    'component': workflow.anomaly_event.component,
                    'severity': workflow.anomaly_event.severity,
                    'timestamp': workflow.anomaly_event.timestamp.isoformat(),
                    'trigger_type': workflow.anomaly_event.trigger_type.value,
                    'anomaly_data': workflow.anomaly_event.anomaly_data,
                    'recommendations': workflow.anomaly_event.recommendations,
                    'rca_result': workflow.anomaly_event.rca_result
                },
                'selected_playbook': workflow.selected_playbook,
                'playbook_description': workflow.playbook_description,
                'playbook_execution': workflow.playbook_execution,
                'current_step_index': workflow.current_step_index,
                'steps': [
                    {
                        'step_id': step.step_id,
                        'agent': step.agent,
                        'description': step.description,
                        'status': step.status.value,
                        'started_at': step.started_at.isoformat() if step.started_at else None,
                        'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                        'duration_seconds': step.duration_seconds,
                        'output': step.output,
                        'error': step.error
                    }
                    for step in workflow.steps
                ],
                'agent_interactions': [
                    {
                        'from_agent': interaction.from_agent,
                        'to_agent': interaction.to_agent,
                        'message_type': interaction.message_type,
                        'timestamp': interaction.timestamp.isoformat(),
                        'status': interaction.status
                    }
                    for interaction in workflow.agent_interactions
                ]
            }
            
            return web.json_response(workflow_detail)
            
        except Exception as e:
            logger.error(f"Error in workflow detail handler: {str(e)}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def trigger_handler(self, request):
        """Manually trigger an anomaly event"""
        try:
            component = request.match_info['component']
            severity = request.query.get('severity', 'HIGH')
            
            if component.upper() not in ['AMF', 'SMF', 'UPF']:
                return web.json_response({"error": "Invalid component"}, status=400)
            
            if self.noc_monitor:
                event = await self.noc_monitor.trigger_manual_event(component, severity)
                return web.json_response({
                    "event_id": event.event_id,
                    "component": event.component,
                    "severity": event.severity,
                    "timestamp": event.timestamp.isoformat()
                })
            else:
                return web.json_response({"error": "Monitor not initialized"}, status=500)
                
        except Exception as e:
            logger.error(f"Error in trigger handler: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def speed_handler(self, request):
        """Set processing speed"""
        try:
            speed_param = request.match_info['speed']
            
            # Parse speed (support 1x, 5x, 10x, 100x format)
            if speed_param.endswith('x'):
                speed = float(speed_param[:-1])
            else:
                speed = float(speed_param)
            
            if speed not in [1.0, 5.0, 10.0, 100.0]:
                return web.json_response({"error": "Invalid speed. Must be 1, 5, 10, or 100"}, status=400)
            
            if self.noc_monitor:
                self.noc_monitor.set_processing_speed(speed)
                return web.json_response({
                    "speed": speed,
                    "message": f"Processing speed set to {speed}x",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return web.json_response({"error": "Monitor not initialized"}, status=500)
                
        except Exception as e:
            logger.error(f"Error in speed handler: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def health_handler(self, request):
        """Health check endpoint"""
        try:
            return web.json_response({
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0",
                "autonomous_mode": True,
                "agents_active": 4
            })
        except Exception as e:
            logger.error(f"Error in health handler: {str(e)}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)
    
    async def initialize(self, data_path: str = "processed_data", playbook_dir: str = "playbooks"):
        """Initialize the enhanced NOC system"""
        try:
            self.stats['start_time'] = datetime.now()
            
            # Initialize NOC data monitor
            self.noc_monitor = NOCDataMonitor(data_path)
            self.noc_monitor.add_event_callback(self._handle_anomaly_event)
            
            # Initialize Ansible executor
            self.ansible_executor = AnsiblePlaybookExecutor(playbook_dir)
            
            logger.info("🚀 Enhanced NOC Dashboard System Initialized")
            logger.info(f"   Data path: {data_path}")
            logger.info(f"   Playbook directory: {playbook_dir}")
            logger.info(f"   Autonomous mode: ACTIVE")
            logger.info(f"   Enhanced agent topology: ENABLED")
            logger.info(f"   Real-time metrics: ENABLED")
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            raise
    
    async def start_monitoring(self):
        """Start NOC data monitoring"""
        try:
            if self.noc_monitor:
                await self.noc_monitor.start_monitoring()
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    async def stop_monitoring(self):
        """Stop NOC data monitoring"""
        try:
            if self.noc_monitor:
                await self.noc_monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
    
    async def _handle_anomaly_event(self, event: AnomalyEvent):
        """Enhanced anomaly event handler with workflow management and topology integration"""
        try:
            self.stats['anomalies_processed'] += 1
            
            # Track anomaly in metrics
            self.metrics_tracker.add_anomaly(event.component, event.timestamp, event.severity)
            
            logger.info(f"🔄 CREATING WORKFLOW: {event.event_id}")
            
            # Create workflow
            workflow = self.workflow_manager.create_workflow(event)
            
            # Start workflow
            self.workflow_manager.start_workflow(workflow.workflow_id)
            
            # Log agent communications and decisions in topology manager
            self.topology_manager.log_communication('diagnostic', 'planning', 'anomaly_report')
            self.topology_manager.log_decision('diagnostic', 'escalate', f'{event.severity} severity anomaly detected in {event.component}')
            
            # Simulate agent workflow with workflow tracking
            await self._simulate_enhanced_agent_workflow(workflow)
            
            # Determine appropriate playbook
            playbook_name = self._select_playbook(event)
            
            if playbook_name:
                workflow.selected_playbook = playbook_name
                workflow.playbook_description = self.ansible_executor.get_playbook_description(playbook_name)
                
                # Log planning decision
                self.topology_manager.log_decision('planning', 'select_playbook', f'Selected {playbook_name} for {event.component} remediation')
                
                # Prepare execution variables
                extra_vars = {
                    "anomaly_id": event.event_id,
                    "component": event.component.lower(),
                    "severity": event.severity,
                    "timestamp": event.timestamp.isoformat(),
                    "autonomous_mode": True,
                    "workflow_id": workflow.workflow_id
                }
                
                # Log execution communication
                self.topology_manager.log_communication('planning', 'execution', 'playbook_execution_request')
                
                # Execute playbook autonomously
                execution_result = await self.ansible_executor.execute_playbook(
                    playbook_name, extra_vars
                )
                
                # Store execution result in workflow
                workflow.playbook_execution = execution_result
                
                # Update statistics
                self.stats['playbooks_executed'] += 1
                if execution_result['status'] == 'success':
                    self.stats['successful_executions'] += 1
                    workflow.success_rate = 100.0
                    self.topology_manager.log_decision('execution', 'success', f'Successfully executed {playbook_name}')
                else:
                    self.stats['failed_executions'] += 1
                    workflow.success_rate = 0.0
                    self.topology_manager.log_decision('execution', 'failure', f'Failed to execute {playbook_name}')
                
                # Store execution history
                self.execution_history.append({
                    "event": event,
                    "execution": execution_result,
                    "workflow_id": workflow.workflow_id,
                    "timestamp": datetime.now()
                })
                
                # Complete execution step
                self.workflow_manager.advance_workflow_step(
                    workflow.workflow_id, 
                    "execution",
                    output=f"Executed {playbook_name} - {execution_result['status']}",
                    error=execution_result.get('error')
                )
                
                # Log validation communication
                self.topology_manager.log_communication('execution', 'validation', 'validation_request')
                
                # Complete validation step
                self.workflow_manager.advance_workflow_step(
                    workflow.workflow_id,
                    "validation", 
                    output=f"Validation completed - Success rate: {workflow.success_rate}%"
                )
                
                # Log validation decision
                self.topology_manager.log_decision('validation', 'complete', f'Workflow validation completed with {workflow.success_rate}% success rate')
                
                # Update agent states
                for agent in self.agent_states:
                    self.agent_states[agent]['tasks_completed'] += 1
                
                logger.info(f"✅ WORKFLOW COMPLETED: {workflow.workflow_id}")
                
            else:
                logger.warning(f"⚠️  No suitable remediation found for: {event.event_id}")
                self.workflow_manager.advance_workflow_step(
                    workflow.workflow_id,
                    "execution",
                    error="No suitable playbook found"
                )
                
        except Exception as e:
            logger.error(f"❌ Error in enhanced workflow processing {event.event_id}: {str(e)}")
            self.stats['failed_executions'] += 1
    
    async def _simulate_enhanced_agent_workflow(self, workflow: WorkflowExecution):
        """Enhanced agent workflow simulation with detailed tracking and topology integration"""
        try:
            current_time = datetime.now()
            
            # Step 1: Diagnostic Agent processes the anomaly
            self.agent_states['diagnostic']['status'] = 'processing'
            self.agent_states['diagnostic']['current_task'] = f'Analyzing {workflow.anomaly_event.component} anomaly'
            
            # Advance to diagnostic step
            self.workflow_manager.advance_workflow_step(
                workflow.workflow_id,
                "diagnostic",
                output=f"Anomaly analysis completed for {workflow.anomaly_event.component}"
            )
            
            interaction1 = AgentInteraction(
                from_agent='diagnostic',
                to_agent='planning',
                message_type='anomaly_report',
                timestamp=current_time,
                status='active'
            )
            self.agent_interactions.append(interaction1)
            workflow.agent_interactions.append(interaction1)
            
            await asyncio.sleep(2)
            
            # Step 2: Planning Agent creates remediation plan
            self.agent_states['diagnostic']['status'] = 'active'
            self.agent_states['diagnostic']['current_task'] = 'Monitoring telemetry'
            self.agent_states['planning']['status'] = 'processing'
            self.agent_states['planning']['current_task'] = f'Creating plan for {workflow.anomaly_event.component}'
            
            # Advance to planning step
            self.workflow_manager.advance_workflow_step(
                workflow.workflow_id,
                "planning",
                output=f"Remediation plan created for {workflow.anomaly_event.component}"
            )
            
            interaction1.status = 'completed'
            interaction2 = AgentInteraction(
                from_agent='planning',
                to_agent='execution',
                message_type='remediation_plan',
                timestamp=current_time + timedelta(seconds=2),
                status='active'
            )
            self.agent_interactions.append(interaction2)
            workflow.agent_interactions.append(interaction2)
            
            await asyncio.sleep(2)
            
            # Steps 3 & 4 will be completed in _handle_anomaly_event after playbook execution
            
        except Exception as e:
            logger.error(f"Error in enhanced agent workflow simulation: {str(e)}")
    
    def _select_playbook(self, event: AnomalyEvent) -> Optional[str]:
        """Select appropriate playbook based on event characteristics"""
        try:
            component = event.component.lower()
            recommendations = str(event.recommendations).lower() if event.recommendations else ""
            
            logger.info(f"🎯 AUTONOMOUS PLAYBOOK SELECTION for {component}")
            
            # Intelligent component-based selection
            if component == "amf":
                if "cpu" in recommendations or "memory" in recommendations:
                    return "scale_amf_resources.yml"
                elif "restart" in recommendations:
                    return "restart_amf_service.yml"
            elif component == "smf":
                if "restart" in recommendations or "service" in recommendations:
                    return "restart_smf_service.yml"
                elif "resource" in recommendations:
                    return "resource_optimization.yml"
            elif component == "upf":
                if "latency" in recommendations or "load" in recommendations:
                    return "adjust_upf_load_balancing.yml"
                elif "throughput" in recommendations:
                    return "resource_optimization.yml"
            
            # Intelligent fallback based on severity
            if event.severity in ["CRITICAL", "HIGH"]:
                return "resource_optimization.yml"
            
            return "resource_optimization.yml"  # Safe default
            
        except Exception as e:
            logger.error(f"Error in playbook selection: {str(e)}")
            return "resource_optimization.yml"

async def main():
    """Main entry point for Enhanced NOC Dashboard with Complete Agent Topology"""
    
    parser = argparse.ArgumentParser(description="Enhanced 5G NOC Dashboard with Advanced Agent Topology")
    parser.add_argument("--data-path", default="processed_data", help="Path to NOC processed data")
    parser.add_argument("--playbook-dir", default="playbooks", help="Directory for Ansible playbooks")
    parser.add_argument("--dashboard-port", type=int, default=30080, help="Dashboard server port")
    parser.add_argument("--dashboard-host", default="0.0.0.0", help="Dashboard server host")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Data polling interval in seconds")
    args = parser.parse_args()
    
    logger.info("🚀 Starting Enhanced 5G NOC Dashboard with Advanced Agent Topology")
    logger.info("=" * 80)
    
    # Initialize enhanced NOC dashboard
    noc_dashboard = EnhancedNOCDashboard()
    
    try:
        await noc_dashboard.initialize(args.data_path, args.playbook_dir)
        
        # Start NOC data monitoring
        await noc_dashboard.start_monitoring()
        
        logger.info(f"🌐 Enhanced Dashboard starting on {args.dashboard_host}:{args.dashboard_port}")
        logger.info(f"🤖 Autonomous agents monitoring: {args.data_path}")
        logger.info(f"📋 Remediation playbooks: {args.playbook_dir}")
        logger.info(f"⏱️  Processing interval: {args.poll_interval} seconds")
        logger.info("=" * 80)
        logger.info(f"🌍 Enhanced NOC Dashboard: http://localhost:{args.dashboard_port}")
        logger.info("🚨 AUTONOMOUS MODE ACTIVE - Advanced Agent Topology")
        logger.info("   Features: Real-time Performance Analytics + Predictive Intelligence")
        logger.info("   Enhanced: Interactive Agent Monitoring + Decision Analytics")
        logger.info("   NEW: Real-time Metric Visualization with Anomaly Tracking")
        logger.info("   Press Ctrl+C to exit")
        
        # Start web server
        runner = web.AppRunner(noc_dashboard.app)
        await runner.setup()
        site = web.TCPSite(runner, args.dashboard_host, args.dashboard_port)
        await site.start()
        
        logger.info("✅ Enhanced NOC Dashboard with Advanced Agent Topology operational")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down Enhanced NOC Dashboard...")
    except Exception as e:
        logger.error(f"❌ Critical error: {str(e)}")
    finally:
        # Cleanup
        try:
            await noc_dashboard.stop_monitoring()
            logger.info("🏁 Enhanced NOC Dashboard stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Enhanced NOC Dashboard shutdown complete")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")