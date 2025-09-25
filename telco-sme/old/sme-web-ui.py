"""
OpenShift AI Hosted Model Serving - Telco AIX - SME Chat UI
Author: Fatih E. NAR
"""

import gradio as gr
import requests
import json
from typing import List, Tuple, Optional, Dict, Any, Generator
from datetime import datetime, timedelta
import io
import base64
from dataclasses import dataclass
import urllib3
import traceback
import time
import threading
from urllib.parse import urljoin
import os
import uuid
import pickle
from pathlib import Path
from collections import deque, defaultdict
import re

# Plotting imports for metrics visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Plotly not available. Metrics will be displayed as text.")

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
@dataclass
class Config:
    """Enhanced configuration for the chat application"""
    api_endpoint: str = "https://put-your-model-route-here"
    model_name: str = "put-your-model-name-here"
    default_temperature: float = 0.2
    default_max_tokens: int = 4096
    admin_username: str = "admin-user"
    admin_password: str = "admin-pwd"
    verify_ssl: bool = False
    
    # Token Authentication
    api_token: str = "put-your-token-here"
    use_token_auth: bool = True
    
    # Timeout settings
    connect_timeout: int = 45
    read_timeout: int = 240  # 4 minutes for non-streaming
    streaming_timeout: int = 600  # 10 minutes for streaming
    
    # Context management
    auto_stream_threshold: int = 4000  # Auto-enable streaming for large contexts
    max_file_chars: int = 3500  # Limit file content size
    max_retry_attempts: int = 5

# Load system prompts from external file
def load_system_prompts():
    """Load system prompts from external JSON file"""
    try:
        prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.json")
    except NameError:
        # Handle case when __file__ is not defined (e.g., interactive mode)
        prompts_file = "system_prompts.json"
    
    # Default compact prompts as fallback
    default_prompts = {
        "Default Assistant": "You are a helpful AI assistant. Provide direct, clear responses without showing your reasoning process.",
        "Technical Expert": "You are a technical expert in software engineering, cloud architecture, and AI/ML. Provide detailed, accurate responses directly without showing internal reasoning.",
        "Code Assistant": "You are an expert programmer. Write clean, well-documented code with explanations and best practices. Give direct answers without showing thinking process.",
        "Data Analyst": "You are a data analyst expert. Help analyze data, create insights, and explain statistical concepts clearly. Provide direct responses.",
        "Creative Writer": "You are a creative writing assistant. Help with storytelling, creative content, and engaging narratives. Give direct creative responses.",
        "Network Expert": "You are a network architect expert. Provide network design, troubleshooting, and optimization guidance.",
        "Telco Expert": "You are a telecommunications expert with expertise in 5G/4G/3G, RAN, Core networks, and telco standards. Provide technical responses with vendor comparisons.",
        "Custom": ""
    }
    
    try:
        if os.path.exists(prompts_file):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                loaded_prompts = json.load(f)
                print(f"✅ Loaded {len(loaded_prompts)} system prompts from {prompts_file}")
                return loaded_prompts
        else:
            print(f"⚠️ System prompts file not found at {prompts_file}, using defaults")
            return default_prompts
    except Exception as e:
        print(f"❌ Error loading system prompts: {e}, using defaults")
        return default_prompts

# Load system prompts at module level
SYSTEM_PROMPTS = load_system_prompts()

class SessionManager:
    """Manages persistent chat sessions"""
    
    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.session_timeout = 24 * 60 * 60  # 24 hours in seconds
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())[:8]  # Short session ID
        self._save_session(session_id, {
            'history': [],
            'created_at': time.time(),
            'last_accessed': time.time(),
            'settings': {
                'system_prompt': 'Default Assistant',
                'custom_prompt': '',
                'temperature': 0.3,
                'max_tokens': 4192
            }
        })
        print(f"📂 Created new session: {session_id}")
        return session_id
    
    def load_session(self, session_id: str) -> dict:
        """Load session data or create new if doesn't exist"""
        if not session_id:
            return self._create_empty_session()
        
        session_file = self.sessions_dir / f"session_{session_id}.pkl"
        
        if not session_file.exists():
            print(f"⚠️ Session {session_id} not found, creating new")
            return self._create_empty_session()
        
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            # Check if session expired
            if time.time() - session_data.get('last_accessed', 0) > self.session_timeout:
                print(f"⏰ Session {session_id} expired, creating new")
                session_file.unlink(missing_ok=True)  # Delete expired session
                return self._create_empty_session()
            
            # Update last accessed time
            session_data['last_accessed'] = time.time()
            self._save_session(session_id, session_data)
            
            print(f"📂 Loaded session: {session_id} ({len(session_data['history'])} messages)")
            return session_data
            
        except Exception as e:
            print(f"❌ Error loading session {session_id}: {e}")
            return self._create_empty_session()
    
    def save_session(self, session_id: str, history: list, settings: dict = None) -> None:
        """Save session data"""
        if not session_id:
            return
        
        session_data = {
            'history': history,
            'last_accessed': time.time(),
            'settings': settings or {}
        }
        
        # Load existing session to preserve created_at
        existing = self.load_session(session_id)
        if 'created_at' in existing:
            session_data['created_at'] = existing['created_at']
        else:
            session_data['created_at'] = time.time()
        
        self._save_session(session_id, session_data)
    
    def _save_session(self, session_id: str, session_data: dict) -> None:
        """Internal method to save session data"""
        session_file = self.sessions_dir / f"session_{session_id}.pkl"
        try:
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
        except Exception as e:
            print(f"❌ Error saving session {session_id}: {e}")
    
    def _create_empty_session(self) -> dict:
        """Create empty session data structure"""
        return {
            'history': [],
            'created_at': time.time(),
            'last_accessed': time.time(),
            'settings': {
                'system_prompt': 'Default Assistant',
                'custom_prompt': '',
                'temperature': 0.3,
                'max_tokens': 4192
            }
        }
    
    def list_sessions(self) -> list:
        """List all active sessions"""
        sessions = []
        for session_file in self.sessions_dir.glob("session_*.pkl"):
            try:
                session_id = session_file.stem.replace("session_", "")
                with open(session_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Skip expired sessions
                if time.time() - data.get('last_accessed', 0) > self.session_timeout:
                    session_file.unlink(missing_ok=True)
                    continue
                
                sessions.append({
                    'id': session_id,
                    'messages': len(data['history']),
                    'created': data.get('created_at', 0),
                    'accessed': data.get('last_accessed', 0)
                })
            except:
                continue
        
        return sorted(sessions, key=lambda x: x['accessed'], reverse=True)
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count"""
        cleaned = 0
        for session_file in self.sessions_dir.glob("session_*.pkl"):
            try:
                with open(session_file, 'rb') as f:
                    data = pickle.load(f)
                
                if time.time() - data.get('last_accessed', 0) > self.session_timeout:
                    session_file.unlink(missing_ok=True)
                    cleaned += 1
            except:
                session_file.unlink(missing_ok=True)
                cleaned += 1
        
        return cleaned

class MetricsCollector:
    """Time series metrics collector with visualization support and persistent storage"""
    
    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.metrics_data = defaultdict(lambda: deque(maxlen=max_points))
        self.timestamps = deque(maxlen=max_points)
        self.pull_interval = 30  # Default 30 seconds
        self.collection_active = False
        self.collection_thread = None
        self.lock = threading.Lock()
        
        # Archive settings
        self.archive_dir = Path("metrics_archive")
        self.archive_dir.mkdir(exist_ok=True)
        self.archive_file = self.archive_dir / "metrics_data.json"
        self.last_save_time = time.time()
        self.save_interval = 60  # Save every 60 seconds
        
        # Load existing data on startup
        self.load_archive()
        
        # vLLM Metric categories optimized for platform teams and users
        # Organized by operational impact and actionability
        self.metric_categories = {
            'system_health': {  # Core System Health - Critical for Operations
                'color': '#FF6B6B',  # Red - Critical alerts
                'display_name': '🏥 System Health',
                'description': 'Request queue, processing state, and system stability',
                'priority': 1,  # Highest priority for platform teams
                'patterns': [
                    r'vllm:num_requests_running',  # Number of requests currently running on GPU
                    r'vllm:num_requests_waiting',  # Number of requests waiting to be processed
                    r'vllm:num_requests_swapped',  # Number of requests swapped to CPU
                    r'vllm:num_preemptions_total',  # Cumulative number of preemptions
                    r'vllm:request_success_total(?!.*created).*',  # Count of successfully processed requests
                ]
            },
            'response_times': {  # User Experience - Response Time Analysis
                'color': '#4ECDC4',  # Teal - Performance focus
                'display_name': '⚡ Response Times',
                'description': 'End-to-end latency, processing time, and user experience metrics',
                'priority': 2,  # High priority for user experience
                'patterns': [
                    r'vllm:time_to_first_token_seconds(?!.*created).*',  # Histogram of time to first token
                    r'vllm:time_per_output_token_seconds(?!.*created).*',  # Histogram of time per output token
                    r'vllm:e2e_request_latency_seconds(?!.*created).*',  # End-to-end request latency
                    r'vllm:request_queue_time_seconds(?!.*created).*',  # Time spent in WAITING phase
                    r'vllm:request_inference_time_seconds(?!.*created).*',  # Time spent in RUNNING phase
                    r'vllm:request_prefill_time_seconds(?!.*created).*',  # Time spent in PREFILL phase
                    r'vllm:request_decode_time_seconds(?!.*created).*',  # Time spent in DECODE phase
                    r'vllm:model_forward_time_milliseconds(?!.*created).*',  # Model forward pass time
                    r'vllm:model_execute_time_milliseconds(?!.*created).*',  # Model execute function time
                ]
            },
            'throughput': {  # Capacity Planning - Token Processing & Throughput
                'color': '#45B7D1',  # Blue - Capacity metrics
                'display_name': '🚀 Throughput',
                'description': 'Token processing rates, generation capacity, and system efficiency',
                'priority': 3,  # Important for capacity planning
                'patterns': [
                    r'vllm:prompt_tokens_total',  # Number of prefill tokens processed
                    r'vllm:generation_tokens_total',  # Number of generation tokens processed
                    r'vllm:tokens_total',  # Total prefill + generation tokens processed
                    r'vllm:iteration_tokens_total(?!.*created).*',  # Histogram of tokens per engine step
                    r'vllm:request_prompt_tokens(?!.*created).*',  # Prefill tokens per request
                    r'vllm:request_generation_tokens(?!.*created).*',  # Generation tokens per request
                    r'vllm:avg_prompt_throughput_toks_per_s',  # Average prefill throughput (deprecated)
                    r'vllm:avg_generation_throughput_toks_per_s',  # Average generation throughput (deprecated)
                ]
            },
            'resource_usage': {  # Resource Optimization - Memory & Cache Usage
                'color': '#96CEB4',  # Green - Resource efficiency
                'display_name': '💾 Resource Usage',
                'description': 'Memory consumption, cache efficiency, and resource optimization',
                'priority': 4,  # Important for cost optimization
                'patterns': [
                    r'vllm:gpu_cache_usage_perc',  # GPU KV-cache usage (1 = 100% usage)
                    r'vllm:cpu_cache_usage_perc',  # CPU KV-cache usage (1 = 100% usage)  
                    r'vllm:gpu_prefix_cache_queries_total',  # GPU prefix cache queries
                    r'vllm:gpu_prefix_cache_hits_total',  # GPU prefix cache hits
                    r'vllm:gpu_prefix_cache_hit_rate',  # GPU prefix cache block hit rate
                    r'vllm:cpu_prefix_cache_hit_rate',  # CPU prefix cache block hit rate
                ]
            },
            'request_patterns': {  # Usage Analytics - Request Patterns & Configuration
                'color': '#FFA07A',  # Light Salmon - Analytics
                'display_name': '📊 Request Patterns',
                'description': 'Request characteristics, parameter usage, and workload analysis',
                'priority': 5,  # Useful for workload understanding
                'patterns': [
                    r'vllm:request_params_n(?!.*created).*',  # Histogram of the n request parameter
                    r'vllm:request_params_max_tokens(?!.*created).*',  # Histogram of max_tokens parameter
                    r'vllm:request_max_num_generation_tokens(?!.*created).*',  # Max requested generation tokens
                ]
            },
            'model_features': {  # Advanced Features - LoRA & Model Configuration
                'color': '#9B59B6',  # Purple - Advanced features
                'display_name': '🧠 Model Features',
                'description': 'LoRA adapters, model configuration, and advanced features',
                'priority': 6,  # Lower priority unless using features
                'patterns': [
                    r'vllm:lora_requests_info',  # Running stats on LoRA requests
                    r'vllm:cache_config_info',  # Cache configuration info
                ]
            },
            'optimization': {  # Performance Optimization - Speculative Decoding
                'color': '#E67E22',  # Orange - Optimization features
                'display_name': '⚙️ Optimization',
                'description': 'Speculative decoding efficiency and advanced optimization features',
                'priority': 7,  # Specialized optimization metrics
                'patterns': [
                    r'vllm:spec_decode_draft_acceptance_rate',  # Speculative token acceptance rate
                    r'vllm:spec_decode_efficiency',  # Speculative decoding system efficiency
                    r'vllm:spec_decode_num_accepted_tokens_total',  # Number of accepted tokens
                    r'vllm:spec_decode_num_draft_tokens_total',  # Number of draft tokens
                    r'vllm:spec_decode_num_emitted_tokens_total',  # Number of emitted tokens
                ]
            },
            'http': {  # HTTP/Web server metrics
                'color': '#17a2b8',  # Info blue
                'patterns': [
                    r'http_requests(?!.*created).*',  # HTTP requests (excluding _created)
                    r'http_request_duration(?!.*created).*',  # HTTP request duration (excluding _created)
                    r'http_request_size_bytes(?!.*created).*',  # HTTP request size (excluding _created)
                    r'http_response_size_bytes(?!.*created).*',  # HTTP response size (excluding _created)
                    r'http_(?!.*created).*',  # All HTTP-related metrics (excluding _created)
                ]
            },
            'system': {  # System resource and runtime metrics
                'color': '#6c757d',  # Gray
                'patterns': [
                    r'python_info',  # Python runtime info
                    r'python_.*',  # Python metrics
                    r'system_.*',  # System metrics
                    r'runtime_.*',  # Runtime metrics
                    r'process_.*',  # Process metrics (CPU, memory, FDs)
                ]
            },
        }
    
    def parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format and extract current values"""
        metrics = {}
        
        for line in metrics_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse metric line: metric_name{labels} value [timestamp]
                parts = line.split(' ')
                if len(parts) >= 2:
                    metric_name_with_labels = parts[0]
                    try:
                        value = float(parts[1])
                        
                        # Extract metric name (before any '{' or ' ')
                        metric_name = metric_name_with_labels.split('{')[0]
                        metrics[metric_name] = value
                    except ValueError:
                        continue
                        
        return metrics
    
    def is_static_timestamp_metric(self, metric_name: str, value: float) -> bool:
        """Determine if a metric is a static timestamp that shouldn't be tracked"""
        try:
            # Ensure inputs are correct types
            if not isinstance(metric_name, str) or not isinstance(value, (int, float)):
                return False
                
            # Primary filter: All _created metrics are timestamps
            if metric_name.endswith('_created'):
                return True
            
            # Secondary filter: Known static timestamp patterns
            static_patterns = [
                r'.*_created$',              # All _created metrics
                r'.*_start_time.*',          # Process start times
                r'process_start_time.*',     # Process start timestamps
                r'.*created.*timestamp.*',   # Explicit timestamp metrics
                r'.*_creation_.*',           # Creation time metrics
                r'.*_initialized.*',         # Initialization timestamps
            ]
            
            for pattern in static_patterns:
                if re.search(pattern, metric_name, re.IGNORECASE):
                    return True
            
            # Tertiary filter: Unix timestamp value detection (2024-2026 range)
            # Current Unix timestamp range: 1.7e9 to 1.8e9 (2025 timeframe)
            if 1.6e9 < value < 1.9e9:
                # If it's a suspiciously large number that looks like a timestamp
                # and has timestamp-like naming, filter it
                timestamp_hints = [
                    'time', 'created', 'start', 'init', 'timestamp', 'epoch'
                ]
                name_lower = metric_name.lower()
                if any(hint in name_lower for hint in timestamp_hints):
                    return True
            
            return False
        except (TypeError, AttributeError, ValueError) as e:
            print(f"⚠️ Error in timestamp detection for {metric_name}: {e}")
            return False
    
    def categorize_metric(self, metric_name: str) -> str:
        """Categorize a metric based on its name"""
        for category, info in self.metric_categories.items():
            for pattern in info['patterns']:
                if re.search(pattern, metric_name, re.IGNORECASE):
                    return category
        return 'other'
    
    def debug_categorization(self):
        """Debug categorization of all current metrics"""
        with self.lock:
            categorized = {'memory': [], 'transactions': [], 'tokens': [], 'model': [], 'other': []}
            
            print(f"\n🔍 CATEGORIZATION DEBUG:")
            print(f"Total metrics to categorize: {len(self.metrics_data)}")
            
            for metric_name in self.metrics_data.keys():
                category = self.categorize_metric(metric_name)
                categorized[category].append(metric_name)
            
            for category, metrics in categorized.items():
                print(f"  {category.upper()}: {len(metrics)} metrics")
                for metric in metrics[:3]:  # Show first 3 examples
                    print(f"    - {metric}")
                if len(metrics) > 3:
                    print(f"    ... and {len(metrics) - 3} more")
            
            return categorized
    
    def add_metrics_data(self, metrics_text: str):
        """Add new metrics data point"""
        if not metrics_text:
            return
            
        with self.lock:
            timestamp = datetime.now()
            parsed_metrics = self.parse_prometheus_metrics(metrics_text)
            
            self.timestamps.append(timestamp)
            
            # Filter out static timestamp metrics and other noise
            dynamic_metrics = {}
            static_count = 0
            
            for metric_name, value in parsed_metrics.items():
                # Ensure metric_name is a string and value is a number
                if not isinstance(metric_name, str) or not isinstance(value, (int, float)):
                    continue
                    
                # Enhanced filtering for timestamp and noise metrics
                try:
                    if (self.is_static_timestamp_metric(metric_name, value) or 
                        metric_name.endswith('_created') or
                        'start_time_seconds' in metric_name or
                        (value > 1.6e9 and any(hint in metric_name.lower() for hint in ['time', 'created', 'start']))):
                        static_count += 1
                        continue
                except (TypeError, AttributeError) as e:
                    print(f"⚠️ Error filtering metric {metric_name}: {e}")
                    continue
                    
                dynamic_metrics[metric_name] = value
                self.metrics_data[metric_name].append(value)
            
            if static_count > 0:
                print(f"🚫 Filtered out {static_count} static timestamp metrics, kept {len(dynamic_metrics)} dynamic metrics")
    
    def get_metrics_by_category(self, category: str) -> Dict[str, List]:
        """Get time series data for a specific category"""
        with self.lock:
            category_metrics = {}
            timestamps = list(self.timestamps)
            
            for metric_name, values in self.metrics_data.items():
                metric_category = self.categorize_metric(metric_name)
                if metric_category == category:
                    category_metrics[metric_name] = {
                        'values': list(values),
                        'timestamps': timestamps[-len(values):] if values else timestamps[:len(values)] if timestamps else []
                    }
            return category_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a performance-focused summary of key metrics for model monitoring"""
        with self.lock:
            if not self.metrics_data:
                return {"error": "No metrics data available"}
            
            # Get latest values for key performance indicators
            latest_metrics = {}
            try:
                for metric_name, values in self.metrics_data.items():
                    if values and isinstance(values, (list, deque)) and len(values) > 0:
                        latest_metrics[metric_name] = values[-1]
            except Exception as e:
                print(f"⚠️ Error processing metrics data: {e}")
                return {"error": f"Error processing metrics: {str(e)}"}
            
            # Calculate performance indicators
            performance_summary = {
                "model_health": {},
                "performance": {},
                "resource_usage": {},
                "request_stats": {},
                "computed_metrics": {}
            }
            
            # Model Health Indicators - Core system state metrics
            if 'vllm:num_requests_running' in latest_metrics:
                performance_summary["model_health"]["requests_running"] = int(latest_metrics['vllm:num_requests_running'])
            if 'vllm:num_requests_waiting' in latest_metrics:
                performance_summary["model_health"]["requests_waiting"] = int(latest_metrics['vllm:num_requests_waiting'])
            if 'vllm:num_requests_swapped' in latest_metrics:
                performance_summary["model_health"]["requests_swapped"] = int(latest_metrics['vllm:num_requests_swapped'])
            if 'vllm:gpu_cache_usage_perc' in latest_metrics:
                # Handle both 0-1 scale and percentage scale values
                cache_val = latest_metrics['vllm:gpu_cache_usage_perc']
                cache_usage = cache_val * 100 if cache_val <= 1.0 else cache_val
                performance_summary["model_health"]["gpu_cache_usage_pct"] = round(cache_usage, 2)
            
            # Performance Metrics
            if 'vllm:prompt_tokens_total' in latest_metrics:
                performance_summary["performance"]["total_prompt_tokens"] = int(latest_metrics['vllm:prompt_tokens_total'])
            if 'vllm:generation_tokens_total' in latest_metrics:
                performance_summary["performance"]["total_generation_tokens"] = int(latest_metrics['vllm:generation_tokens_total'])
            
            # Request Statistics
            if 'vllm:request_success_total' in latest_metrics:
                performance_summary["request_stats"]["successful_requests"] = int(latest_metrics['vllm:request_success_total'])
            if 'vllm:num_preemptions_total' in latest_metrics:
                performance_summary["request_stats"]["preemptions"] = int(latest_metrics['vllm:num_preemptions_total'])
            
            # Latency Metrics (from histograms)
            latency_metrics = {}
            for metric_name, value in latest_metrics.items():
                if 'latency_seconds_sum' in metric_name:
                    count_metric = metric_name.replace('_sum', '_count')
                    if count_metric in latest_metrics and latest_metrics[count_metric] > 0:
                        avg_latency = value / latest_metrics[count_metric]
                        metric_type = metric_name.replace('vllm:', '').replace('_seconds_sum', '')
                        latency_metrics[metric_type] = round(avg_latency * 1000, 2)  # Convert to ms
            
            performance_summary["performance"]["avg_latencies_ms"] = latency_metrics
            
            # Resource Usage
            if 'process_resident_memory_bytes' in latest_metrics:
                memory_gb = latest_metrics['process_resident_memory_bytes'] / (1024**3)
                performance_summary["resource_usage"]["memory_usage_gb"] = round(memory_gb, 2)
            if 'process_cpu_seconds_total' in latest_metrics:
                performance_summary["resource_usage"]["cpu_seconds_total"] = round(latest_metrics['process_cpu_seconds_total'], 2)
            if 'process_open_fds' in latest_metrics:
                performance_summary["resource_usage"]["open_file_descriptors"] = int(latest_metrics['process_open_fds'])
            
            # Computed Performance Metrics
            computed = {}
            
            # Cache hit rate
            if ('vllm:gpu_prefix_cache_hits_total' in latest_metrics and 
                'vllm:gpu_prefix_cache_queries_total' in latest_metrics and
                latest_metrics['vllm:gpu_prefix_cache_queries_total'] > 0):
                hits = latest_metrics['vllm:gpu_prefix_cache_hits_total']
                queries = latest_metrics['vllm:gpu_prefix_cache_queries_total']
                hit_rate = (hits / queries) * 100
                computed["cache_hit_rate_pct"] = round(hit_rate, 2)
            
            # Total tokens processed
            prompt_tokens = latest_metrics.get('vllm:prompt_tokens_total', 0)
            gen_tokens = latest_metrics.get('vllm:generation_tokens_total', 0)
            computed["total_tokens_processed"] = int(prompt_tokens + gen_tokens)
            
            # Request completion rate and throughput calculations
            if len(self.timestamps) >= 2:
                time_span_minutes = (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 60
                if time_span_minutes > 0:
                    success_count = latest_metrics.get('vllm:request_success_total', 0)
                    computed["requests_per_minute"] = round(success_count / time_span_minutes, 2)
                    
                    # Preemption rate
                    preemptions = latest_metrics.get('vllm:num_preemptions_total', 0)
                    computed["preemption_rate"] = round(preemptions / time_span_minutes, 2)
            
            # Average time-to-first-token (TTFT) calculation
            if ('vllm:time_to_first_token_seconds_sum' in latest_metrics and 
                'vllm:time_to_first_token_seconds_count' in latest_metrics and
                latest_metrics['vllm:time_to_first_token_seconds_count'] > 0):
                ttft_sum = latest_metrics['vllm:time_to_first_token_seconds_sum']
                ttft_count = latest_metrics['vllm:time_to_first_token_seconds_count']
                computed["avg_ttft_seconds"] = round(ttft_sum / ttft_count, 3)
            
            # Success rate calculation
            total_requests = latest_metrics.get('vllm:request_success_total', 0)
            failed_requests = 0  # vLLM doesn't expose failure count directly
            if total_requests > 0:
                # Assume high success rate if no failures tracked
                computed["success_rate_pct"] = 99.5  # Default assumption
            
            # Cache efficiency score (0-100 based on hit rate and usage)
            cache_hit_rate = computed.get("cache_hit_rate_pct", 0)
            cache_usage = performance_summary["model_health"].get("gpu_cache_usage_pct", 0)
            
            # Efficiency scoring: balance between hit rate and usage
            if cache_hit_rate > 0 and cache_usage > 0:
                # High hit rate is good, moderate usage (60-80%) is optimal
                optimal_usage = 70  # 70% is considered optimal
                usage_efficiency = 100 - abs(cache_usage - optimal_usage) * 2
                usage_efficiency = max(0, min(100, usage_efficiency))
                
                # Combine hit rate and usage efficiency
                computed["cache_efficiency_score"] = round((cache_hit_rate + usage_efficiency) / 2, 1)
            else:
                computed["cache_efficiency_score"] = 0
            
            performance_summary["computed_metrics"] = computed
            
            return performance_summary
    
    def format_performance_summary_text(self) -> str:
        """Format performance summary as readable text for display"""
        summary = self.get_performance_summary()
        
        if "error" in summary:
            return f"❌ {summary['error']}"
        
        lines = []
        lines.append("🚀 **MODEL PERFORMANCE DASHBOARD**")
        lines.append("=" * 50)
        
        # Model Health Status
        if summary.get("model_health"):
            lines.append("\n🟢 **MODEL HEALTH**")
            health = summary["model_health"]
            
            if "requests_running" in health:
                lines.append(f"   🔄 Active Requests: {health['requests_running']}")
            if "requests_waiting" in health:
                status = "🟡" if health['requests_waiting'] > 0 else "🟢"
                lines.append(f"   {status} Queued Requests: {health['requests_waiting']}")
            if "gpu_cache_usage_pct" in health:
                usage = health['gpu_cache_usage_pct']
                status = "🔴" if usage > 80 else "🟡" if usage > 60 else "🟢"
                lines.append(f"   {status} GPU Cache Usage: {usage:.1f}%")
        
        # Performance Metrics
        if summary.get("performance"):
            lines.append("\n📊 **THROUGHPUT & PERFORMANCE**")
            perf = summary["performance"]
            
            if "total_prompt_tokens" in perf:
                lines.append(f"   📝 Prompt Tokens Processed: {perf['total_prompt_tokens']:,}")
            if "total_generation_tokens" in perf:
                lines.append(f"   🎯 Generation Tokens: {perf['total_generation_tokens']:,}")
            
            if "avg_latencies_ms" in perf and perf["avg_latencies_ms"]:
                lines.append("   ⚡ Average Latencies:")
                for latency_type, ms in perf["avg_latencies_ms"].items():
                    status = "🔴" if ms > 5000 else "🟡" if ms > 1000 else "🟢"
                    readable_name = latency_type.replace('_', ' ').title()
                    lines.append(f"      {status} {readable_name}: {ms:.1f}ms")
        
        # Request Statistics
        if summary.get("request_stats"):
            lines.append("\n📈 **REQUEST STATISTICS**")
            stats = summary["request_stats"]
            
            if "successful_requests" in stats:
                lines.append(f"   ✅ Successful Requests: {stats['successful_requests']:,}")
            if "preemptions" in stats:
                status = "🔴" if stats['preemptions'] > 10 else "🟡" if stats['preemptions'] > 0 else "🟢"
                lines.append(f"   {status} Preemptions: {stats['preemptions']:,}")
        
        # Computed Metrics
        if summary.get("computed_metrics"):
            lines.append("\n🧮 **COMPUTED METRICS**")
            computed = summary["computed_metrics"]
            
            if "cache_hit_rate_pct" in computed:
                rate = computed['cache_hit_rate_pct']
                status = "🟢" if rate > 80 else "🟡" if rate > 50 else "🔴"
                lines.append(f"   {status} Cache Hit Rate: {rate:.1f}%")
            
            if "total_tokens_processed" in computed:
                lines.append(f"   🔢 Total Tokens: {computed['total_tokens_processed']:,}")
            
            if "requests_per_minute" in computed:
                rpm = computed['requests_per_minute']
                status = "🟢" if rpm > 5 else "🟡" if rpm > 1 else "🔴"
                lines.append(f"   {status} Request Rate: {rpm:.2f}/min")
        
        # Resource Usage
        if summary.get("resource_usage"):
            lines.append("\n💾 **RESOURCE USAGE**")
            resources = summary["resource_usage"]
            
            if "memory_usage_gb" in resources:
                mem = resources['memory_usage_gb']
                status = "🔴" if mem > 16 else "🟡" if mem > 8 else "🟢"
                lines.append(f"   {status} Memory Usage: {mem:.2f} GB")
            
            if "cpu_seconds_total" in resources:
                lines.append(f"   ⚙️ CPU Time: {resources['cpu_seconds_total']:.1f}s")
            
            if "open_file_descriptors" in resources:
                fds = resources['open_file_descriptors']
                status = "🔴" if fds > 1000 else "🟡" if fds > 500 else "🟢"
                lines.append(f"   {status} Open FDs: {fds:,}")
        
        lines.append("\n" + "=" * 50)
        lines.append("📅 Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return "\n".join(lines)
    
    def analyze_metric_units(self, metric_name: str, values: list) -> dict:
        """Analyze metric to determine optimal visualization type"""
        name_lower = metric_name.lower()
        max_val = max(values) if values else 0
        min_val = min(values) if values else 0
        
        # Unit type detection
        if 'bytes' in name_lower or 'memory' in name_lower:
            return {'unit': 'bytes', 'viz_type': 'gauge', 'format': 'memory'}
        elif any(x in name_lower for x in ['seconds', 'latency', 'time', 'duration']):
            return {'unit': 'seconds', 'viz_type': 'histogram', 'format': 'time'}
        elif 'tokens' in name_lower and 'per_second' not in name_lower:
            return {'unit': 'count', 'viz_type': 'bar', 'format': 'integer'}
        elif any(x in name_lower for x in ['per_second', 'rate', 'throughput']):
            return {'unit': 'rate', 'viz_type': 'speedometer', 'format': 'decimal'}
        elif 'perc' in name_lower or (max_val <= 100 and min_val >= 0):
            return {'unit': 'percentage', 'viz_type': 'radial', 'format': 'percent'}
        elif 'active' in name_lower or 'running' in name_lower or 'waiting' in name_lower:
            return {'unit': 'current', 'viz_type': 'line', 'format': 'integer'}
        else:
            return {'unit': 'generic', 'viz_type': 'line', 'format': 'decimal'}
    
    def auto_group_metrics(self, metrics_data: dict) -> dict:
        """Group metrics by unit type for optimal visualization"""
        groups = {
            'memory_gauges': {'metrics': [], 'title': 'Memory Usage', 'viz_type': 'gauge_grid'},
            'latency_dist': {'metrics': [], 'title': 'Latency Distribution', 'viz_type': 'histogram_grid'},
            'token_counts': {'metrics': [], 'title': 'Token Volumes', 'viz_type': 'bar_chart'},
            'throughput': {'metrics': [], 'title': 'Rates & Throughput', 'viz_type': 'speedometer_grid'},
            'percentages': {'metrics': [], 'title': 'Utilization %', 'viz_type': 'radial_grid'},
            'active_status': {'metrics': [], 'title': 'Live Counters', 'viz_type': 'line_chart'}
        }
        
        for metric_name, data in metrics_data.items():
            values = data.get('values', [])
            if not values:
                continue
                
            analysis = self.analyze_metric_units(metric_name, values)
            
            # Route to appropriate group based on unit analysis
            if analysis['unit'] == 'bytes':
                groups['memory_gauges']['metrics'].append((metric_name, data, analysis))
            elif analysis['unit'] == 'seconds':
                groups['latency_dist']['metrics'].append((metric_name, data, analysis))
            elif analysis['unit'] == 'count' and 'tokens' in metric_name.lower():
                groups['token_counts']['metrics'].append((metric_name, data, analysis))
            elif analysis['unit'] == 'rate':
                groups['throughput']['metrics'].append((metric_name, data, analysis))
            elif analysis['unit'] == 'percentage':
                groups['percentages']['metrics'].append((metric_name, data, analysis))
            elif analysis['unit'] == 'current':
                groups['active_status']['metrics'].append((metric_name, data, analysis))
            else:
                groups['active_status']['metrics'].append((metric_name, data, analysis))
        
        # Filter out empty groups
        return {k: v for k, v in groups.items() if v['metrics']}
    
    def create_memory_gauge_grid(self, metrics: list) -> go.Figure:
        """Create gauge grid for memory metrics"""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        rows = (len(metrics) + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}] for _ in range(rows)],
            subplot_titles=[name for name, _, _ in metrics]
        )
        
        for idx, (metric_name, data, analysis) in enumerate(metrics[:4]):  # Limit to 4 gauges
            values = data['values']
            if not values:
                continue
                
            current_val = values[-1]
            max_val = max(values) * 1.2  # 20% headroom
            
            # Convert bytes to appropriate unit
            if current_val > 1e9:
                display_val = current_val / 1e9
                max_display = max_val / 1e9
                unit = "GB"
            elif current_val > 1e6:
                display_val = current_val / 1e6
                max_display = max_val / 1e6
                unit = "MB"
            else:
                display_val = current_val / 1e3
                max_display = max_val / 1e3
                unit = "KB"
            
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=display_val,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{metric_name.replace('_', ' ').title()}"},
                    number={'suffix': f" {unit}"},
                    gauge={
                        'axis': {'range': [None, max_display]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, max_display*0.5], 'color': "lightgray"},
                            {'range': [max_display*0.5, max_display*0.8], 'color': "yellow"},
                            {'range': [max_display*0.8, max_display], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_display*0.9
                        }
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=300*rows, title="Memory Usage Gauges")
        return fig
    
    def create_throughput_speedometer(self, metrics: list) -> go.Figure:
        """Create speedometer for throughput metrics"""
        from plotly.subplots import make_subplots
        
        rows = (len(metrics) + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}] for _ in range(rows)]
        )
        
        for idx, (metric_name, data, analysis) in enumerate(metrics[:4]):
            values = data['values']
            if not values:
                continue
                
            current_rate = values[-1]
            max_rate = max(values) * 1.5
            
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_rate,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': metric_name.replace('_', ' ').title()},
                    delta={'reference': sum(values)/len(values)},
                    gauge={
                        'axis': {'range': [None, max_rate]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, max_rate*0.3], 'color': "lightgray"},
                            {'range': [max_rate*0.3, max_rate*0.7], 'color': "yellow"},
                            {'range': [max_rate*0.7, max_rate], 'color': "lime"}
                        ]
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=300*rows, title="Throughput Speedometers")
        return fig
    
    def create_radial_progress(self, metrics: list) -> go.Figure:
        """Create radial progress charts for percentages"""
        from plotly.subplots import make_subplots
        
        rows = (len(metrics) + 1) // 2
        fig = make_subplots(
            rows=rows, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}] for _ in range(rows)]
        )
        
        for idx, (metric_name, data, analysis) in enumerate(metrics[:4]):
            values = data['values']
            if not values:
                continue
                
            current_pct = values[-1]
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=current_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': metric_name.replace('_', ' ').title()},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "purple"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ]
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(height=300*rows, title="Utilization Percentages")
        return fig
    
    def format_metric_value(self, metric_name: str, value: float) -> str:
        """Format metric value based on vLLM metric conventions for platform teams"""
        name_lower = metric_name.lower()
        
        # vLLM-specific metric formatting with platform team friendly units
        if name_lower.startswith('vllm:'):
            # Cache usage percentages (0-1 scale converted to percentage)
            if any(x in name_lower for x in ['cache_usage_perc', 'hit_rate', 'acceptance_rate', 'efficiency']):
                return f"{value:.1f}%"
            
            # Request counts (gauges) - Critical system health indicators
            elif any(x in name_lower for x in ['num_requests_running', 'num_requests_waiting', 'num_requests_swapped']):
                return f"{int(value):,} req" if value != 1 else f"{int(value)} req"
            
            # Time metrics in seconds - User experience focused
            elif 'seconds' in name_lower:
                if value > 3600:  # Hours
                    return f"{value/3600:.1f}h"
                elif value > 60:  # Minutes  
                    return f"{value/60:.1f}m"
                elif value > 1:   # Seconds
                    return f"{value:.2f}s"
                elif value > 0.001:  # Milliseconds
                    return f"{value*1000:.0f}ms"
                else:  # Microseconds
                    return f"{value*1000000:.0f}μs"
            
            # Time metrics in milliseconds - Model execution times
            elif 'milliseconds' in name_lower:
                if value > 1000:
                    return f"{value/1000:.2f}s"
                else:
                    return f"{value:.0f}ms"
            
            # Token counts (counters) - Throughput and capacity metrics
            elif 'tokens' in name_lower:
                if value > 1e9:  # Billions
                    return f"{value/1e9:.2f}B tokens"
                elif value > 1e6:  # Millions
                    return f"{value/1e6:.1f}M tokens"
                elif value > 1e3:  # Thousands
                    return f"{value/1e3:.1f}K tokens"
                else:
                    return f"{int(value):,} tokens"
            
            # Success counters and preemptions
            elif any(x in name_lower for x in ['success_total', 'preemptions_total']):
                if value > 1e6:
                    return f"{value/1e6:.1f}M"
                elif value > 1e3:
                    return f"{value/1e3:.1f}K"
                else:
                    return f"{int(value):,}"
            
            # Request parameters (n, max_tokens) - Usually small integers
            elif any(x in name_lower for x in ['request_params_n', 'request_params_max_tokens', 'request_max_num_generation_tokens']):
                return f"{int(value):,}"
            
            # Cache queries/hits (counters)
            elif 'cache' in name_lower and any(x in name_lower for x in ['queries', 'hits']):
                return f"{int(value):,}"
            
            # Info metrics (gauges with static values)
            elif 'info' in name_lower:
                return f"{value:.0f}"
        
        # Memory/bytes formatting with better accuracy
        elif 'bytes' in name_lower or ('memory' in name_lower and 'usage' in name_lower):
            if value >= 1024**3:  # GB (1024^3)
                return f"{value/(1024**3):.2f} GB"
            elif value >= 1024**2:  # MB (1024^2) 
                return f"{value/(1024**2):.2f} MB"
            elif value >= 1024:  # KB
                return f"{value/1024:.2f} KB"
            else:
                return f"{value:.0f} B"
        
        # Process-specific metrics
        elif 'process_start_time' in name_lower:
            # This is usually a Unix timestamp - convert to relative time
            import time
            current_time = time.time()
            uptime = current_time - value
            if uptime > 86400:  # > 1 day
                return f"{uptime/86400:.1f} days ago"
            elif uptime > 3600:  # > 1 hour
                return f"{uptime/3600:.1f} hours ago"
            elif uptime > 60:  # > 1 minute
                return f"{uptime/60:.1f} minutes ago"
            else:
                return f"{uptime:.0f} seconds ago"
        
        elif 'process_cpu_seconds' in name_lower:
            # This is cumulative CPU time
            if value > 3600:
                return f"{value/3600:.2f} CPU-hours"
            elif value > 60:
                return f"{value/60:.2f} CPU-minutes"
            else:
                return f"{value:.2f} CPU-seconds"
        
        elif any(x in name_lower for x in ['open_fds', 'max_fds']):
            # File descriptors - just show as integer
            return f"{int(value):,}"
        
        # Generic time/seconds formatting
        elif any(x in name_lower for x in ['seconds', 'latency', 'time', 'duration']):
            if value > 3600:
                return f"{value/3600:.2f} h"
            elif value > 60:
                return f"{value/60:.2f} m"
            elif value > 1:
                return f"{value:.3f} s"
            else:
                return f"{value*1000:.1f} ms"
        
        # Percentage formatting
        elif 'perc' in name_lower or '_rate' in name_lower:
            return f"{value:.1f}%"
        
        # Rate formatting
        elif 'per_second' in name_lower:
            return f"{value:.2f}/s"
        
        # Integer counts
        elif any(x in name_lower for x in ['count', 'total', 'active', 'running', 'waiting']):
            return f"{int(value):,}"
        
        # Default decimal
        else:
            if value > 1000:
                return f"{value:,.2f}"
            else:
                return f"{value:.3f}"
    
    def create_enhanced_metrics_dashboard(self, category: str) -> str:
        """Create enhanced HTML dashboard with eye-catching visuals for vLLM metrics"""
        try:
            metrics_data = self.get_metrics_by_category(category)
            if not metrics_data:
                return f"<div style='text-align: center; padding: 40px; color: #666;'>No {category} metrics available</div>"
            
            # Get category color
            category_color = self.metric_categories.get(category, {}).get('color', '#666')
            
            # Sort metrics by importance for display
            sorted_metrics = self.sort_metrics_by_importance(metrics_data.items(), category)
            
            # Create dashboard header
            html = f"""
            <div style='background: linear-gradient(135deg, {category_color}20, white); border-radius: 12px; padding: 20px; margin: 10px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
                    <h3 style='margin: 0; color: {category_color}; font-size: 1.4em; font-weight: bold;'>
                        {self.get_category_emoji(category)} {category.title()} Metrics
                    </h3>
                    <div style='background: {category_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold;'>
                        {len(sorted_metrics)} Active Metrics
                    </div>
                </div>
            """
            
            # Add performance summary if applicable
            if category in ['server', 'request']:
                html += self.create_performance_summary(category, sorted_metrics, category_color)
            
            # Create metrics grid with larger cards for better readability
            html += """<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px;'>"""
            
            for metric_name, data in sorted_metrics:
                values = data['values']
                if not values:
                    continue
                
                current = values[-1]
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                sample_count = len(values)
                
                # Create metric card
                html += self.create_metric_card(metric_name, current, min_val, max_val, avg_val, sample_count, category_color)
            
            html += "</div></div>"  # Close grid and main container
            return html
            
        except Exception as e:
            print(f"Error creating enhanced dashboard for {category}: {e}")
            return f"<div style='text-align: center; padding: 40px; color: red;'>Error loading {category} metrics: {str(e)}</div>"
    
    def sort_metrics_by_importance(self, metrics_items, category: str):
        """Sort metrics by importance based on category"""
        importance_order = {
            'scheduler': ['num_requests_running', 'num_requests_waiting', 'num_requests_swapped', 'num_preemptions'],
            'latency': ['time_to_first_token', 'time_per_output_token', 'e2e_request_latency', 'request_queue_time'],
            'throughput': ['prompt_tokens_total', 'generation_tokens_total', 'tokens_total', 'iteration_tokens'],
            'cache': ['gpu_cache_usage_perc', 'gpu_prefix_cache_hit_rate', 'prefix_cache_hits', 'prefix_cache_queries'],
            'parameters': ['request_params_max_tokens', 'request_params_n', 'request_max_num_generation_tokens'],
            'adapters': ['lora_requests_info', 'cache_config_info'],
            'speculative': ['spec_decode_draft_acceptance_rate', 'spec_decode_efficiency'],
            'http': ['http_requests_total', 'http_request_duration'],
            'system': ['python_info', 'process_memory']
        }
        
        priority_keywords = importance_order.get(category, [])
        
        def get_priority(item):
            metric_name, _ = item
            for i, keyword in enumerate(priority_keywords):
                if keyword.lower() in metric_name.lower():
                    return i
            return len(priority_keywords)
        
        return sorted(metrics_items, key=get_priority)
    
    def get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""
        emojis = {
            'scheduler': '🎛️',
            'latency': '⏱️',
            'throughput': '🚀',
            'cache': '💾',
            'parameters': '⚙️',
            'adapters': '🔗',
            'speculative': '🔮',
            'http': '🌐',
            'system': '💻'
        }
        return emojis.get(category, '📊')
    
    def create_performance_summary(self, category: str, metrics, color: str) -> str:
        """Create performance summary panel for key metrics"""
        if category == 'scheduler':
            return self.create_scheduler_summary(metrics, color)
        elif category == 'latency':
            return self.create_latency_summary(metrics, color)
        return ""
    
    def create_scheduler_summary(self, metrics, color: str) -> str:
        """Create vLLM scheduler state summary"""
        running_requests = 0
        waiting_requests = 0
        cache_usage = 0
        total_memory = 0
        
        print(f"🔍 Server summary debug - processing {len(metrics)} metrics")
        
        for metric_name, data in metrics:
            values = data['values']
            if not values:
                continue
            current = values[-1]
            
            print(f"  - {metric_name}: {current}")
            
            # More flexible pattern matching
            metric_lower = metric_name.lower()
            if any(pattern in metric_lower for pattern in ['requests_running', 'running', 'num_requests_running']):
                running_requests = current
                print(f"    → Found running requests: {current}")
            elif any(pattern in metric_lower for pattern in ['requests_waiting', 'waiting', 'num_requests_waiting']):
                waiting_requests = current
                print(f"    → Found waiting requests: {current}")
            elif any(pattern in metric_lower for pattern in ['cache_usage', 'gpu_cache_usage', 'cache.*perc']):
                cache_usage = current
                print(f"    → Found cache usage: {current}")
            elif any(pattern in metric_lower for pattern in ['memory_usage_bytes', 'memory']):
                total_memory = current
                print(f"    → Found memory usage: {current}")
        
        total_requests = running_requests + waiting_requests
        status_color = '#28a745' if total_requests < 10 else '#ffc107' if total_requests < 50 else '#dc3545'
        
        # Format memory display
        memory_display = "N/A"
        if total_memory > 0:
            if total_memory > 1024*1024*1024:  # GB
                memory_display = f"{total_memory/(1024*1024*1024):.1f} GB"
            elif total_memory > 1024*1024:  # MB
                memory_display = f"{total_memory/(1024*1024):.1f} MB"
            elif total_memory > 1024:  # KB
                memory_display = f"{total_memory/1024:.1f} KB"
            else:
                memory_display = f"{total_memory:.0f} B"
        
        print(f"🔍 Final summary values: requests={total_requests}, running={running_requests}, waiting={waiting_requests}, cache={cache_usage}%, memory={memory_display}")
        
        return f"""
        <div style='background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; border-left: 4px solid {color};'>
            <div style='display: flex; justify-content: space-around; text-align: center;'>
                <div>
                    <div style='font-size: 2em; font-weight: bold; color: {status_color};'>{int(total_requests)}</div>
                    <div style='color: #666; font-size: 0.9em;'>Total Requests</div>
                </div>
                <div>
                    <div style='font-size: 2em; font-weight: bold; color: #17a2b8;'>{int(running_requests)}</div>
                    <div style='color: #666; font-size: 0.9em;'>Running</div>
                </div>
                <div>
                    <div style='font-size: 2em; font-weight: bold; color: #6c757d;'>{int(waiting_requests)}</div>
                    <div style='color: #666; font-size: 0.9em;'>Waiting</div>
                </div>
                <div>
                    <div style='font-size: 1.8em; font-weight: bold; color: {color};'>{cache_usage:.1f}%</div>
                    <div style='color: #666; font-size: 0.9em;'>Cache Usage</div>
                </div>
                <div>
                    <div style='font-size: 1.6em; font-weight: bold; color: #28a745;'>{memory_display}</div>
                    <div style='color: #666; font-size: 0.9em;'>Memory</div>
                </div>
            </div>
        </div>
        """
    
    def create_latency_summary(self, metrics, color: str) -> str:
        """Create vLLM latency performance summary"""
        ttft = 0
        latency = 0
        queue_time = 0
        
        for metric_name, data in metrics:
            values = data['values']
            if not values:
                continue
            avg_val = sum(values) / len(values)
            
            if 'time_to_first_token' in metric_name:
                ttft = avg_val
            elif 'e2e_request_latency' in metric_name:
                latency = avg_val
            elif 'queue_time' in metric_name:
                queue_time = avg_val
        
        return f"""
        <div style='background: white; border-radius: 8px; padding: 15px; margin-bottom: 20px; border-left: 4px solid {color};'>
            <div style='display: flex; justify-content: space-around; text-align: center;'>
                <div>
                    <div style='font-size: 1.8em; font-weight: bold; color: {color};'>{ttft:.2f}s</div>
                    <div style='color: #666; font-size: 0.9em;'>Avg TTFT</div>
                </div>
                <div>
                    <div style='font-size: 1.8em; font-weight: bold; color: #17a2b8;'>{latency:.2f}s</div>
                    <div style='color: #666; font-size: 0.9em;'>Avg Latency</div>
                </div>
                <div>
                    <div style='font-size: 1.8em; font-weight: bold; color: #6c757d;'>{queue_time:.2f}s</div>
                    <div style='color: #666; font-size: 0.9em;'>Avg Queue Time</div>
                </div>
            </div>
        </div>
        """
    
    def create_metric_card(self, metric_name: str, current: float, min_val: float, max_val: float, avg_val: float, sample_count: int, color: str) -> str:
        """Create individual metric card with gauge visualization"""
        # Format values with error handling
        try:
            current_str = self.format_metric_value(metric_name, current) if current is not None else "N/A"
            min_str = self.format_metric_value(metric_name, min_val) if min_val is not None else "N/A"
            max_str = self.format_metric_value(metric_name, max_val) if max_val is not None else "N/A"
            avg_str = self.format_metric_value(metric_name, avg_val) if avg_val is not None else "N/A"
        except Exception as e:
            print(f"⚠️ Error formatting {metric_name}: {e}")
            current_str = str(current) if current is not None else "Error"
            min_str = str(min_val) if min_val is not None else "Error"
            max_str = str(max_val) if max_val is not None else "Error" 
            avg_str = str(avg_val) if avg_val is not None else "Error"
        
        # Calculate trend color based on current vs average with safety checks
        try:
            if current is not None and avg_val is not None and avg_val != 0:
                if current > avg_val * 1.1:
                    trend_color = '#dc3545'  # Red for high values
                    trend_icon = '📈'
                elif current < avg_val * 0.9:
                    trend_color = '#28a745'  # Green for low values
                    trend_icon = '📉'
                else:
                    trend_color = '#ffc107'  # Yellow for stable
                    trend_icon = '📊'
            else:
                trend_color = '#6c757d'  # Gray for unknown
                trend_icon = '➖'
        except:
            trend_color = '#6c757d'
            trend_icon = '❓'
        
        # Calculate gauge percentage (0-100%) with safety checks
        try:
            if max_val is not None and min_val is not None and current is not None and max_val > min_val:
                gauge_percent = min(100, max(0, ((current - min_val) / (max_val - min_val)) * 100))
            else:
                gauge_percent = 50
        except:
            gauge_percent = 50
        
        # Clean metric name for display with smart truncation
        display_name = self.create_smart_display_name(metric_name)
        
        return f"""
        <div style='background: white; border-radius: 8px; padding: 15px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' title='{metric_name}'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;'>
                <div style='font-weight: bold; color: #333; font-size: 0.9em; line-height: 1.2; flex: 1; margin-right: 8px;'>{display_name}</div>
                <div style='color: {trend_color}; font-size: 1.1em; flex-shrink: 0;'>{trend_icon}</div>
            </div>
            
            <div style='margin: 10px 0;'>
                <div style='background: #f0f0f0; border-radius: 10px; height: 8px; overflow: hidden;'>
                    <div style='background: linear-gradient(90deg, {color}, {color}80); height: 100%; width: {gauge_percent}%; transition: width 0.3s ease;'></div>
                </div>
            </div>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85em;'>
                <div>
                    <div style='color: {trend_color}; font-weight: bold; font-size: 1.1em;'>{current_str}</div>
                    <div style='color: #666;'>Current</div>
                </div>
                <div>
                    <div style='color: #333; font-weight: bold;'>{avg_str}</div>
                    <div style='color: #666;'>Average</div>
                </div>
                <div>
                    <div style='color: #28a745; font-weight: bold;'>{min_str}</div>
                    <div style='color: #666;'>Min</div>
                </div>
                <div>
                    <div style='color: #dc3545; font-weight: bold;'>{max_str}</div>
                    <div style='color: #666;'>Max</div>
                </div>
            </div>
            
            <div style='text-align: center; margin-top: 8px; color: #888; font-size: 0.8em;'>
                {sample_count} samples
            </div>
        </div>
        """
    
    def create_smart_display_name(self, metric_name: str) -> str:
        """Create smart, distinguishable display names for metrics"""
        # Remove vllm prefix and common prefixes
        name = metric_name.replace('vllm:', '').replace('process_', '')
        
        # Smart abbreviations and replacements for common patterns
        replacements = {
            'request_': 'Req ',
            'generation_': 'Gen ',
            'prompt_': 'Prompt ',
            'tokens_': 'Tokens ',
            'seconds': 'Sec',
            'total': 'Total',
            'latency': 'Lat',
            'duration': 'Dur',
            'time_to_first_token': 'TTFT',
            'time_per_output_token': 'TPOT',
            'e2e_request_latency': 'E2E Lat',
            'queue_time': 'Queue',
            'prefill_time': 'Prefill',
            'decode_time': 'Decode',
            'inference_time': 'Inference',
            'http_request_duration': 'HTTP Dur',
            'max_num_generation': 'Max Gen',
            'iteration_': 'Iter ',
            'cpu_seconds': 'CPU Sec',
            'memory_bytes': 'Memory',
            'open_fds': 'Open FDs',
            'max_fds': 'Max FDs',
            'start_time': 'Start Time',
            'cache_hits': 'Cache Hits',
            'cache_queries': 'Cache Queries',
            'gpu_prefix_cache': 'GPU Cache',
            'prefix_cache': 'PCache',
            # Additional patterns from the debug screenshots
            'request_prompt_tokens': 'Req Prompt Tokens',
            'request_generation_tokens': 'Req Gen Tokens', 
            'request_max_num_generation_tokens': 'Req Max Gen Tokens',
            'request_inference_time': 'Req Inference Time',
            'request_prefill_time': 'Req Prefill Time',
            'request_decode_time': 'Req Decode Time',
            'request_queue_time': 'Req Queue Time',
            'iteration_tokens': 'Iter Tokens',
            'completion_tokens': 'Completion Tokens',
        }
        
        # Apply replacements
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Clean up underscores and format
        name = name.replace('_', ' ').strip()
        
        # Handle histogram bucket indicators
        if 'bucket' in name.lower():
            # Extract bucket value for histogram metrics
            import re
            bucket_match = re.search(r'le="([^"]+)"', metric_name)
            if bucket_match:
                bucket_val = bucket_match.group(1)
                name = name.split(' bucket')[0] + f' ≤{bucket_val}'
        
        # Handle sum/count suffixes for histogram metrics  
        if name.endswith(' sum'):
            name = name.replace(' sum', ' (Sum)')
        elif name.endswith(' count'):
            name = name.replace(' count', ' (Count)')
            
        # Smart capitalization
        words = name.split()
        capitalized_words = []
        for word in words:
            if word.upper() in ['TTFT', 'TPOT', 'E2E', 'HTTP', 'GPU', 'CPU', 'FDS']:
                capitalized_words.append(word.upper())
            elif word.lower() in ['sec', 'lat', 'dur', 'gen', 'req']:
                capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word.title())
        
        name = ' '.join(capitalized_words)
        
        # Final length management - be more generous but still readable
        if len(name) > 35:
            # Try to truncate at word boundaries
            words = name.split()
            truncated = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= 32:  # +1 for space
                    truncated.append(word)
                    current_length += len(word) + 1
                else:
                    break
            
            if len(truncated) > 0:
                name = ' '.join(truncated) + '...'
            else:
                # Fallback: hard truncate
                name = name[:32] + '...'
        
        return name
    
    def create_metrics_table(self, category: str) -> str:
        """Legacy method - redirect to enhanced dashboard"""
        return self.create_enhanced_metrics_dashboard(category)
    
    def create_time_series_plot(self, category: str) -> Optional[str]:
        """Return metrics table instead of plot"""
        return self.create_metrics_table(category)
    
    
    def set_pull_interval(self, seconds: int):
        """Set the metrics collection interval"""
        self.pull_interval = max(5, seconds)  # Minimum 5 seconds
    
    def start_collection(self, client):
        """Start automated metrics collection"""
        if self.collection_active:
            return
            
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, 
            args=(client,), 
            daemon=True
        )
        self.collection_thread.start()
    
    def stop_collection(self):
        """Stop automated metrics collection"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _collection_loop(self, client):
        """Background thread for collecting metrics"""
        while self.collection_active:
            try:
                metrics_result = client.get_metrics()
                if metrics_result.get('success'):
                    self.add_metrics_data(metrics_result.get('data', ''))
                    
                    # Periodically save to archive
                    current_time = time.time()
                    if current_time - self.last_save_time > self.save_interval:
                        self.save_archive()
                        self.last_save_time = current_time
                        
            except Exception as e:
                print(f"⚠️ Metrics collection error: {e}")
                
            time.sleep(self.pull_interval)
        
        # Save final state when collection stops
        print("💾 Collection stopped, saving final archive...")
        self.save_archive(force=True)
    
    def save_archive(self, force=False):
        """Save current metrics data to persistent storage"""
        try:
            # Check cooldown to prevent excessive saves (unless forced)
            current_time = time.time()
            if not force and current_time - self.last_save_time < 5.0:  # 5 second minimum cooldown
                return  # Skip save if too recent
                
            with self.lock:
                # Convert deque objects to lists for JSON serialization
                archive_data = {
                    'metrics_data': {
                        name: list(values) for name, values in self.metrics_data.items()
                    },
                    'timestamps': [ts.isoformat() for ts in self.timestamps],
                    'last_updated': datetime.now().isoformat(),
                    'pull_interval': self.pull_interval,
                    'max_points': self.max_points
                }
                
                # Create backup of existing archive
                if self.archive_file.exists():
                    backup_file = self.archive_dir / f"metrics_backup_{int(time.time())}.json"
                    self.archive_file.rename(backup_file)
                    
                    # Keep only last 5 backups
                    backups = sorted(self.archive_dir.glob("metrics_backup_*.json"))
                    for old_backup in backups[:-5]:
                        old_backup.unlink(missing_ok=True)
                
                # Save current data
                with open(self.archive_file, 'w') as f:
                    json.dump(archive_data, f, indent=2)
                
                print(f"📦 Metrics archive saved: {len(self.metrics_data)} metrics, {len(self.timestamps)} timestamps")
                
                # Update last save time after successful save
                self.last_save_time = current_time
                
        except Exception as e:
            print(f"❌ Error saving metrics archive: {str(e)}")
    
    def load_archive(self):
        """Load metrics data from persistent storage"""
        try:
            if not self.archive_file.exists():
                print("📦 No metrics archive found, starting fresh")
                return
            
            with open(self.archive_file, 'r') as f:
                archive_data = json.load(f)
            
            # Load metrics data
            for metric_name, values in archive_data.get('metrics_data', {}).items():
                self.metrics_data[metric_name] = deque(values, maxlen=self.max_points)
            
            # Load timestamps
            timestamp_strings = archive_data.get('timestamps', [])
            self.timestamps = deque([
                datetime.fromisoformat(ts) for ts in timestamp_strings
            ], maxlen=self.max_points)
            
            # Update settings from archive
            self.pull_interval = archive_data.get('pull_interval', self.pull_interval)
            
            last_updated = archive_data.get('last_updated')
            loaded_count = len(self.metrics_data)
            timestamps_count = len(self.timestamps)
            
            print(f"✅ Metrics archive loaded: {loaded_count} metrics, {timestamps_count} timestamps")
            if last_updated:
                print(f"📅 Last updated: {last_updated}")
                
        except Exception as e:
            print(f"❌ Error loading metrics archive: {str(e)}")
            print("📦 Starting with empty metrics collection")
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics data to a file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_export_{timestamp}.json"
            
            export_path = self.archive_dir / filename
            
            with self.lock:
                export_data = {
                    'exported_at': datetime.now().isoformat(),
                    'export_version': '1.0',
                    'metrics_count': len(self.metrics_data),
                    'timestamps_count': len(self.timestamps),
                    'pull_interval': self.pull_interval,
                    'metrics_data': {
                        name: list(values) for name, values in self.metrics_data.items()
                    },
                    'timestamps': [ts.isoformat() for ts in self.timestamps],
                    'categories': list(self.metric_categories.keys())
                }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return f"✅ Metrics exported to {export_path}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def import_metrics(self, filename: str) -> str:
        """Import metrics data from a file"""
        try:
            import_path = self.archive_dir / filename
            
            if not import_path.exists():
                return f"❌ File not found: {filename}"
            
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Merge with existing data
            with self.lock:
                for metric_name, values in import_data.get('metrics_data', {}).items():
                    # Merge values, keeping newest points
                    existing_values = list(self.metrics_data[metric_name])
                    combined_values = existing_values + values
                    self.metrics_data[metric_name] = deque(combined_values[-self.max_points:], maxlen=self.max_points)
                
                # Merge timestamps
                import_timestamps = [
                    datetime.fromisoformat(ts) for ts in import_data.get('timestamps', [])
                ]
                existing_timestamps = list(self.timestamps)
                combined_timestamps = existing_timestamps + import_timestamps
                # Sort by time and take most recent points
                combined_timestamps.sort()
                self.timestamps = deque(combined_timestamps[-self.max_points:], maxlen=self.max_points)
            
            # Save merged data
            self.save_archive(force=True)
            
            imported_count = len(import_data.get('metrics_data', {}))
            return f"✅ Imported {imported_count} metrics from {filename}"
            
        except Exception as e:
            return f"❌ Import failed: {str(e)}"
    
    def test_metric_patterns(self, sample_metrics: list = None) -> dict:
        """Test metric categorization with sample data or provided metrics"""
        if sample_metrics is None:
            # Sample metrics from the provided data
            sample_metrics = [
                'memory_usage_bytes', 'http_requests_total', 'http_request_duration_seconds',
                'prompt_tokens_total', 'completion_tokens_total', 'inference_time_seconds',
                'gpu_memory_usage', 'active_requests', 'tokens_per_second', 'model_memory_usage',
                'vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:gpu_cache_usage_perc',
                'vllm:gpu_prefix_cache_queries_total', 'vllm:gpu_prefix_cache_hits_total',
                'vllm:time_to_first_token_seconds_bucket', 'vllm:time_per_output_token_seconds_count',
                'vllm:e2e_request_latency_seconds_sum', 'vllm:request_params_n_bucket',
                'vllm:request_params_max_tokens_count', 'vllm:iteration_tokens_total_bucket',
                'vllm:request_success_total', 'vllm:num_preemptions_total',
                'process_cpu_seconds_total', 'process_resident_memory_bytes', 'python_gc_objects_collected_total'
            ]
        
        results = {'categorized': {}, 'unmatched': []}
        
        print(f"\n🧪 TESTING METRIC CATEGORIZATION:")
        print(f"Testing {len(sample_metrics)} metrics...\n")
        
        for metric_name in sample_metrics:
            category = self.categorize_metric(metric_name)
            if category not in results['categorized']:
                results['categorized'][category] = []
            results['categorized'][category].append(metric_name)
            
            if category == 'other':
                results['unmatched'].append(metric_name)
        
        print(f"\n📊 CATEGORIZATION RESULTS:")
        for category, metrics in results['categorized'].items():
            print(f"  {category.upper()}: {len(metrics)} metrics")
            for metric in metrics[:3]:  # Show first 3
                print(f"    ✓ {metric}")
            if len(metrics) > 3:
                print(f"    ... and {len(metrics) - 3} more")
        
        if results['unmatched']:
            print(f"\n❌ UNMATCHED METRICS ({len(results['unmatched'])}):")
            for metric in results['unmatched']:
                print(f"    ✗ {metric}")
        else:
            print(f"\n✅ ALL METRICS SUCCESSFULLY CATEGORIZED!")
        
        return results

class StreamingResponse:
    """Handle streaming response accumulation"""
    def __init__(self):
        self.content = ""
        self.is_complete = False
        self.error = None

class ChatClient:
    """Enhanced chat client with streaming and timeout handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create optimized session with proper settings"""
        session = requests.Session()
        session.verify = self.config.verify_ssl
        
        # Build headers
        headers = {
            'User-Agent': 'OpenShift-AI-Chat/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=120, max=100'
        }
        
        # Add token authentication if enabled
        if self.config.use_token_auth and self.config.api_token:
            headers['Authorization'] = f'Bearer {self.config.api_token}'
            print("🔐 Token authentication enabled")
        
        # Optimize connection settings
        session.headers.update(headers)
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # We handle retries manually
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def test_connection(self) -> Dict[str, Any]:
        """Comprehensive connection diagnostics"""
        results = {}
        
        # Test 1: Basic health check
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/health", 
                timeout=self.config.connect_timeout
            )
            results['health'] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'response': response.text[:200],
                'latency': response.elapsed.total_seconds()
            }
        except Exception as e:
            results['health'] = {'success': False, 'error': str(e)}
        
        # Test 2: Available models
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/v1/models", 
                timeout=self.config.connect_timeout
            )
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                results['models'] = {
                    'success': True,
                    'available': available_models,
                    'configured': self.config.model_name,
                    'match': self.config.model_name in available_models
                }
            else:
                results['models'] = {
                    'success': False, 
                    'status': response.status_code,
                    'response': response.text[:200]
                }
        except Exception as e:
            results['models'] = {'success': False, 'error': str(e)}
        
        # Test 3: Simple chat completion (non-streaming)
        try:
            test_payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.1,
                "stream": False
            }
            start_time = time.time()
            response = self.session.post(
                f"{self.config.api_endpoint}/v1/chat/completions",
                json=test_payload,
                timeout=self.config.connect_timeout
            )
            latency = time.time() - start_time
            
            results['chat_test'] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'latency': latency,
                'response': response.text[:200] if response.status_code != 200 else "OK"
            }
        except Exception as e:
            results['chat_test'] = {'success': False, 'error': str(e)}
        
        # Test 4: Streaming test
        try:
            test_payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Count to 3"}],
                "max_tokens": 20,
                "temperature": 0.1,
                "stream": True
            }
            start_time = time.time()
            response = self.session.post(
                f"{self.config.api_endpoint}/v1/chat/completions",
                json=test_payload,
                stream=True,
                timeout=(self.config.connect_timeout, 60)
            )
            
            if response.status_code == 200:
                # Test streaming response
                content = ""
                for line in response.iter_lines():
                    if line and len(content) < 50:  # Limit test
                        line = line.decode('utf-8')
                        if line.startswith("data: ") and line[6:] != "[DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                if 'choices' in chunk:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content += delta['content']
                            except:
                                continue
                        if content:  # Got some content, test passes
                            break
                
                latency = time.time() - start_time
                results['streaming_test'] = {
                    'success': True,
                    'latency': latency,
                    'content_received': len(content) > 0
                }
            else:
                results['streaming_test'] = {
                    'success': False,
                    'status': response.status_code,
                    'response': response.text[:200]
                }
        except Exception as e:
            results['streaming_test'] = {'success': False, 'error': str(e)}
        
        return results
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get API version information"""
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/version", 
                timeout=5  # Short timeout for management endpoints
            )
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                }
            else:
                return {'success': False, 'status': response.status_code, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/metrics", 
                timeout=5  # Short timeout for management endpoints
            )
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.text  # Metrics are usually in Prometheus format
                }
            else:
                return {'success': False, 'status': response.status_code, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Get OpenAPI specification"""
        try:
            response = self.session.get(
                f"{self.config.api_endpoint}/openapi.json", 
                timeout=5  # Short timeout for management endpoints
            )
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': response.json()
                }
            else:
                return {'success': False, 'status': response.status_code, 'error': response.text}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _estimate_context_size(self, messages: List[Dict[str, str]]) -> int:
        """Estimate total context size in characters"""
        return sum(len(msg.get('content', '')) for msg in messages)
    
    def _should_use_streaming(self, messages: List[Dict[str, str]], max_tokens: int) -> bool:
        """Determine if streaming should be used based on context size"""
        context_size = self._estimate_context_size(messages)
        
        # Use streaming for:
        # 1. Large contexts (>4000 chars)
        # 2. Large max_tokens (>1000)
        # 3. Multiple history messages (>3)
        return (
            context_size > self.config.auto_stream_threshold or
            max_tokens > 1000 or
            len(messages) > 4
        )
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.6,
        max_tokens: int = 2048,
        force_streaming: bool = False
    ) -> str:
        """Smart chat completion with fallback to non-streaming"""
        
        # Decide on streaming
        use_streaming = force_streaming or self._should_use_streaming(messages, max_tokens)
        context_size = self._estimate_context_size(messages)
        
        print(f"🧠 Context size: {context_size} chars, Streaming: {use_streaming}")
        
        if use_streaming:
            print("🌊 Trying streaming first...")
            streaming_result = self._stream_completion(messages, temperature, max_tokens)
            
            # If streaming failed but didn't return an error message, try direct
            if streaming_result.startswith("❌") and context_size < 8000:
                print("🔄 Streaming failed, trying direct completion as fallback...")
                return self._direct_completion(messages, temperature, min(max_tokens, 1000))
            else:
                return streaming_result
        else:
            return self._direct_completion(messages, temperature, max_tokens)
    
    def _direct_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> str:
        """Direct completion for small contexts"""
        url = f"{self.config.api_endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        print(f"🔄 Direct request to: {url}")
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                timeout = (self.config.connect_timeout, self.config.read_timeout)
                response = self.session.post(url, json=payload, timeout=timeout)
                
                print(f"📡 Response: {response.status_code}")
                
                if response.status_code == 504:
                    print("⏰ Gateway timeout - switching to streaming")
                    return self._stream_completion(messages, temperature, max_tokens)
                elif response.status_code != 200:
                    error_detail = response.text[:300]
                    if attempt < self.config.max_retry_attempts - 1:
                        print(f"❌ Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return f"API Error {response.status_code}: {error_detail}"
                
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print(f"✅ Success! Length: {len(content)}")
                    return content
                else:
                    return f"Unexpected response format: {result}"
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retry_attempts - 1:
                    print("🔄 Retrying with streaming...")
                    return self._stream_completion(messages, temperature, max_tokens)
                else:
                    return "❌ Request timed out. Try reducing context size or message length."
            except Exception as e:
                if attempt < self.config.max_retry_attempts - 1:
                    print(f"💥 Attempt {attempt + 1} error: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return f"❌ Error: {str(e)}"
        
        return "❌ All retry attempts failed"
    
    def _stream_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float, 
        max_tokens: int
    ) -> str:
        """Debug-enhanced streaming completion with detailed logging"""
        url = f"{self.config.api_endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        print(f"🌊 DEBUG: Streaming request to: {url}")
        print(f"📦 DEBUG: Payload keys: {list(payload.keys())}")
        
        for attempt in range(self.config.max_retry_attempts):
            accumulated = ""
            try:
                print(f"🚀 DEBUG: Attempt {attempt + 1} starting...")
                
                response = self.session.post(
                    url, 
                    json=payload, 
                    stream=True, 
                    timeout=(self.config.connect_timeout, None)
                )
                
                print(f"📡 DEBUG: Response status: {response.status_code}")
                print(f"📋 DEBUG: Response headers: {dict(response.headers)}")
                
                if response.status_code != 200:
                    error_detail = response.text[:500]
                    print(f"❌ DEBUG: Error response: {error_detail}")
                    if attempt < self.config.max_retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return f"❌ Streaming Error {response.status_code}: {error_detail}"
                
                # Process streaming response with extensive debugging
                start_time = time.time()
                chunk_count = 0
                line_count = 0
                data_lines = 0
                content_chunks = 0
                
                print("🌊 DEBUG: Starting to process streaming response...")
                
                try:
                    for line in response.iter_lines():
                        line_count += 1
                        current_time = time.time()
                        
                        # Timeout protection
                        if current_time - start_time > self.config.streaming_timeout:
                            print(f"⏰ DEBUG: Timeout after {self.config.streaming_timeout}s")
                            if accumulated:
                                return accumulated + "\n\n[Truncated: timeout]"
                            break
                        
                        if line:
                            # Decode line
                            try:
                                line_str = line.decode('utf-8')
                            except UnicodeDecodeError:
                                print(f"⚠️ DEBUG: Unicode decode error for line {line_count}")
                                continue
                            
                            # Log every 100th line for debugging
                            if line_count % 100 == 0:
                                print(f"🔍 DEBUG: Processed {line_count} lines, {data_lines} data lines, {content_chunks} content chunks")
                            
                            # Process data lines
                            if line_str.startswith("data: "):
                                data_lines += 1
                                data = line_str[6:].strip()
                                
                                # Debug first few data lines
                                if data_lines <= 5:
                                    print(f"📄 DEBUG: Data line {data_lines}: {data[:100]}...")
                                
                                if data == "[DONE]":
                                    print("✅ DEBUG: Received [DONE] signal")
                                    break
                                    
                                if data and data != "":
                                    try:
                                        chunk = json.loads(data)
                                        chunk_count += 1
                                        
                                        # Debug chunk structure
                                        if chunk_count <= 3:
                                            print(f"🧩 DEBUG: Chunk {chunk_count} keys: {list(chunk.keys())}")
                                            if 'choices' in chunk:
                                                print(f"🧩 DEBUG: Choices[0] keys: {list(chunk['choices'][0].keys())}")
                                        
                                        if 'choices' in chunk and len(chunk['choices']) > 0:
                                            choice = chunk['choices'][0]
                                            delta = choice.get('delta', {})
                                            
                                            if 'content' in delta:
                                                content_piece = delta.get('content', '')
                                                if content_piece:  # Only add non-empty content
                                                    accumulated += content_piece
                                                    content_chunks += 1
                                                    
                                                    # Progress every 500 chars
                                                    if len(accumulated) % 500 == 0:
                                                        print(f"📝 DEBUG: {len(accumulated)} chars, {content_chunks} content chunks")
                                            
                                            # Check for finish
                                            if choice.get('finish_reason'):
                                                finish_reason = choice.get('finish_reason')
                                                print(f"🏁 DEBUG: Finished with reason: {finish_reason}")
                                                break
                                                
                                    except json.JSONDecodeError as e:
                                        print(f"⚠️ DEBUG: JSON error on line {data_lines}: {str(e)}")
                                        print(f"⚠️ DEBUG: Problematic data: {data[:200]}")
                                        continue
                                    except Exception as e:
                                        print(f"⚠️ DEBUG: Chunk processing error: {str(e)}")
                                        continue
                    
                    # Final debug summary
                    elapsed = time.time() - start_time
                    print(f"🏁 DEBUG: Stream complete after {elapsed:.1f}s")
                    print(f"📊 DEBUG: {line_count} lines, {data_lines} data lines, {chunk_count} chunks, {content_chunks} content chunks")
                    print(f"📏 DEBUG: Final accumulated length: {len(accumulated)}")
                    
                    if accumulated:
                        print(f"✅ DEBUG: SUCCESS! Returning {len(accumulated)} chars")
                        print(f"📖 DEBUG: First 200 chars: {accumulated[:200]}")
                        return accumulated
                    else:
                        print("📭 DEBUG: No content accumulated!")
                        print(f"📊 DEBUG: Had {chunk_count} chunks but {content_chunks} content chunks")
                        
                        if attempt < self.config.max_retry_attempts - 1:
                            print(f"🔄 DEBUG: Retrying attempt {attempt + 2}...")
                            time.sleep(2)
                            continue
                        return f"❌ No content after processing {chunk_count} chunks"
                        
                except Exception as stream_error:
                    print(f"💥 DEBUG: Stream processing exception: {stream_error}")
                    print(f"🔍 DEBUG: Exception traceback: {traceback.format_exc()}")
                    
                    if accumulated:
                        print(f"💾 DEBUG: Returning {len(accumulated)} partial chars")
                        return accumulated + f"\n\n[Stream error: {stream_error}]"
                    
                    if attempt < self.config.max_retry_attempts - 1:
                        continue
                    return f"❌ Stream processing failed: {stream_error}"
                
                finally:
                    try:
                        response.close()
                    except:
                        pass
                    
            except requests.exceptions.Timeout:
                print(f"⏰ DEBUG: Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "❌ Connection timed out"
                
            except Exception as e:
                print(f"💥 DEBUG: Request exception on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"❌ Request failed: {str(e)}"
        
        return f"❌ All {self.config.max_retry_attempts} attempts failed"
    
    def health_check(self) -> bool:
        """Simple health check"""
        try:
            results = self.test_connection()
            return results.get('health', {}).get('success', False)
        except:
            return False

class ChatInterface:
    """Enhanced chat interface with smart processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = ChatClient(config)
        self.system_prompts = SYSTEM_PROMPTS.copy()  # Keep a local copy
        self._processing = False  # Flag to prevent double processing
        self.session_manager = SessionManager()  # Add session management
        self.metrics_collector = MetricsCollector()  # Add metrics collection
    
    def process_message(
        self,
        message: str,
        history: List[List[str]],
        system_prompt: str,
        custom_prompt: str,
        temperature: float,
        max_tokens: int,
        uploaded_file: Optional[Any] = None,
        session_id: str = None
    ) -> Tuple[str, List[List[str]], Optional[Any], str]:
        """Enhanced message processing with UI debugging"""
        
        if not message.strip():
            return "", history, None, session_id or ""
        
        # Create or get session ID
        if not session_id:
            session_id = self.session_manager.create_session()
        
        # Prevent double processing
        if self._processing:
            print("⚠️ Already processing, ignoring duplicate request")
            return "", history, None, session_id
        
        self._processing = True
        
        try:
            print(f"\n{'='*60}")
            print(f"🚀 PROCESSING: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            print(f"🌡️ TEMPERATURE: {temperature}")
            print(f"📏 MAX_TOKENS: {max_tokens}")
            print(f"🎯 SYSTEM_PROMPT: {system_prompt}")
            print(f"{'='*60}")
            
            # Build system prompt
            active_system_prompt = custom_prompt if custom_prompt.strip() else self.system_prompts.get(system_prompt, "")
            
            # Build messages list
            messages = []
            if active_system_prompt:
                messages.append({"role": "system", "content": active_system_prompt})
            
            # Add conversation history (limit to prevent huge contexts)
            recent_history = history[-20:] if len(history) > 20 else history  # Now we get individual messages, so double the limit
            for msg in recent_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # New message format
                    messages.append({"role": msg['role'], "content": msg['content']})
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    # Backward compatibility with old tuple format
                    user_msg, assistant_msg = msg
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})
            
            # Handle file upload with size limits (only for current message)
            if uploaded_file is not None:
                file_content = self._process_file(uploaded_file)
                message = f"{message}\n\n[File content]:\n{file_content}"
                print(f"📎 File attached: {getattr(uploaded_file, 'name', 'unknown')} ({len(file_content)} chars)")
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Context analysis
            context_size = sum(len(msg.get('content', '')) for msg in messages)
            print(f"📊 Context: {context_size} chars, Temp: {temperature}, Tokens: {max_tokens}")
            
            if context_size > 20000:
                error_response = "❌ Context too large. Please start a new conversation or upload a smaller file."
                print(f"❌ Context too large: {context_size} chars")
                new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_response}]
                # Save session with error
                self.session_manager.save_session(session_id, new_history, {
                    'system_prompt': system_prompt,
                    'custom_prompt': custom_prompt,
                    'temperature': temperature,
                    'max_tokens': max_tokens
                })
                print(f"🔄 UI DEBUG: Returning history with {len(new_history)} items")
                return "", new_history, None, session_id
            
            # Process with smart completion
            print("🤖 Calling chat completion...")
            response = self.client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            print(f"✅ Response received: {len(response)} chars")
            print(f"📝 Response preview: {response[:200]}...")
            
            # Ensure response is a string and not empty
            if not isinstance(response, str):
                response = str(response)
            
            # Clean up model reasoning tags
            if response.startswith('<think>') and '</think>' in response:
                # Extract content after thinking tags
                think_end = response.find('</think>')
                if think_end != -1:
                    response = response[think_end + 8:].strip()
                    print(f"🧠 Cleaned thinking tags, new length: {len(response)}")
            
            # Remove only specific problematic HTML-like tags while preserving formatting
            import re
            import json
            # Only remove specific thinking tags and common HTML tags, preserve line breaks
            response = re.sub(r'</?think>', '', response)
            response = re.sub(r'</?reasoning>', '', response)
            response = re.sub(r'</?analysis>', '', response)
            # Clean up but preserve formatting
            response = response.strip()
            
            # If response looks like JSON, try to format it properly
            if response.startswith('{') and response.endswith('}'):
                try:
                    # Parse and reformat JSON with proper indentation
                    parsed_json = json.loads(response)
                    response = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    print("🎨 Formatted JSON response for better readability")
                except (json.JSONDecodeError, ValueError):
                    # If it's not valid JSON, leave it as is
                    pass
            
            if not response.strip():
                response = "❌ Empty response received from model after cleaning"
                print("⚠️ Empty response after cleaning")
            
            print(f"📝 Final cleaned response preview: {response[:200]}...")
            
            # Create new history entry
            new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
            print(f"🔄 UI DEBUG: Creating history entry:")
            print(f"   - User message: {len(message)} chars")
            print(f"   - Assistant response: {len(response)} chars")
            print(f"   - Total history items: {len(new_history)}")
            
            # Validate the history structure
            try:
                if len(new_history) >= 2:
                    user_msg = new_history[-2]
                    assistant_msg = new_history[-1]
                    if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                        print(f"✅ Valid history entry: [{len(user_msg.get('content', ''))} chars user, {len(assistant_msg.get('content', ''))} chars assistant]")
                    else:
                        print(f"⚠️ Invalid history entry format: {user_msg}, {assistant_msg}")
                else:
                    print("⚠️ Not enough history entries to validate")
            except Exception as e:
                print(f"❌ History validation error: {e}")
            
            # Save successful session
            self.session_manager.save_session(session_id, new_history, {
                'system_prompt': system_prompt,
                'custom_prompt': custom_prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            })
            
            print(f"🔄 UI DEBUG: Returning empty message and {len(new_history)} history items")
            print(f"💾 Session {session_id} saved with {len(new_history)} messages")
            return "", new_history, None, session_id
            
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            print(f"💥 Exception: {error_msg}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            
            new_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]
            # Save session with error
            self.session_manager.save_session(session_id, new_history, {
                'system_prompt': system_prompt,
                'custom_prompt': custom_prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            })
            print(f"🔄 UI DEBUG: Error case - returning {len(new_history)} history items")
            return "", new_history, None, session_id
        
        finally:
            self._processing = False
    
    def load_session(self, session_id: str) -> Tuple[List[List[str]], str, str, float, int]:
        """Load session and return history and settings"""
        if not session_id:
            return [], "Default Assistant", "", self.config.default_temperature, self.config.default_max_tokens
        
        session_data = self.session_manager.load_session(session_id)
        settings = session_data.get('settings', {})
        
        return (
            session_data.get('history', []),
            settings.get('system_prompt', 'Default Assistant'),
            settings.get('custom_prompt', ''),
            settings.get('temperature', self.config.default_temperature),
            settings.get('max_tokens', self.config.default_max_tokens)
        )
    
    def new_session(self) -> Tuple[List[List[str]], str]:
        """Create a new session"""
        session_id = self.session_manager.create_session()
        return [], session_id
    
    def get_session_list(self) -> Tuple[str, List[str]]:
        """Get formatted list of sessions and their IDs"""
        sessions = self.session_manager.list_sessions()
        if not sessions:
            return "No active sessions found.", []
        
        session_text = "# 📂 Active Sessions\n\n"
        session_ids = []
        
        for i, session in enumerate(sessions[:10]):  # Show last 10 sessions
            created = datetime.fromtimestamp(session['created']).strftime('%Y-%m-%d %H:%M')
            accessed = datetime.fromtimestamp(session['accessed']).strftime('%Y-%m-%d %H:%M')
            session_text += f"**[{i+1}] Session `{session['id']}`**\n"
            session_text += f"- Messages: {session['messages']}\n"
            session_text += f"- Created: {created}\n"
            session_text += f"- Last accessed: {accessed}\n\n"
            session_ids.append(session['id'])
        
        return session_text, session_ids
    
    def _process_file(self, file) -> str:
        """Process uploaded file with size limits"""
        try:
            if hasattr(file, 'name'):
                file_path = file.name
                file_ext = file_path.lower().split('.')[-1]
                
                if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
                    return f"[Image: {file_path.split('/')[-1]}] Note: Describe what you want to know about this image."
                elif file_ext in ['pdf', 'xlsx', 'docx']:
                    return f"[Binary file: {file_path.split('/')[-1]}] Note: Please convert to text or describe the content."
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Strict size limit to prevent timeouts
                        if len(content) > self.config.max_file_chars:
                            content = content[:self.config.max_file_chars] + f"\n\n[File truncated to {self.config.max_file_chars} chars to prevent timeouts]"
                        return content
            return "Unable to read file"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def clear_history(self) -> List:
        """Clear conversation history"""
        return []
    
    def export_conversation(self, history: List[List[str]]) -> str:
        """Export conversation history"""
        if not history:
            return "No conversation to export."
        
        export_text = f"# Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        msg_count = 0
        for i, msg in enumerate(history):
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # New message format
                if msg['role'] == 'user':
                    msg_count += 1
                    export_text += f"## Message {msg_count}\n\n"
                    export_text += f"**User:** {msg['content']}\n\n"
                elif msg['role'] == 'assistant':
                    export_text += f"**Assistant:** {msg['content']}\n\n"
                    export_text += "---\n\n"
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                # Backward compatibility with old tuple format
                user, assistant = msg
                msg_count += 1
                export_text += f"## Message {msg_count}\n\n"
                export_text += f"**User:** {user}\n\n"
                export_text += f"**Assistant:** {assistant}\n\n"
                export_text += "---\n\n"
        
        return export_text
    
    def test_simple_streaming(self) -> str:
        """Test streaming with a simple request"""
        print("🧪 Testing simple streaming...")
        try:
            test_messages = [{"role": "user", "content": "Count from 1 to 5"}]
            response = self.client._stream_completion(test_messages, 0.1, 50)
            return f"# 🧪 Streaming Test Results\n\n**Response:** {response}\n\n**Length:** {len(response)} characters"
        except Exception as e:
            return f"# 🧪 Streaming Test Results\n\n**Status:** FAILED\n**Error:** {str(e)}"
    
    def test_ui_update(self, history: List[List[str]]) -> Tuple[str, List[List[str]], Optional[Any]]:
        """Test UI update with a sample response"""
        print("🧪 Testing UI update...")
        test_response = "This is a test response to verify UI updates are working correctly. ✅"
        test_message = "UI Test"
        
        new_history = history + [{"role": "user", "content": test_message}, {"role": "assistant", "content": test_response}]
        print(f"🔄 UI TEST: Added entry, total items: {len(new_history)}")
        
        return "", new_history, None
    
    def reload_system_prompts(self) -> str:
        """Reload system prompts from file"""
        try:
            self.system_prompts = load_system_prompts()
            return f"✅ Successfully reloaded {len(self.system_prompts)} system prompts"
        except Exception as e:
            return f"❌ Error reloading prompts: {str(e)}"
    
    def save_custom_prompt(self, prompt_name: str, prompt_content: str) -> str:
        """Save a custom prompt to the file"""
        if not prompt_name or not prompt_content:
            return "❌ Please provide both prompt name and content"
        
        try:
            prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.json")
            
            # Load current prompts
            current_prompts = {}
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    current_prompts = json.load(f)
            
            # Add/update the prompt
            action = "updated" if prompt_name in current_prompts else "created"
            current_prompts[prompt_name] = prompt_content
            
            # Save back to file
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(current_prompts, f, indent=4, ensure_ascii=False)
            
            # Reload prompts
            self.system_prompts = load_system_prompts()
            
            return f"✅ Successfully {action} prompt '{prompt_name}'"
        except Exception as e:
            return f"❌ Error saving prompt: {str(e)}"
    
    def delete_prompt(self, prompt_name: str) -> str:
        """Delete a prompt from the file"""
        if not prompt_name:
            return "❌ Please select a prompt to delete"
        
        if prompt_name in ["Default Assistant", "Technical Expert", "Code Assistant"]:
            return "❌ Cannot delete core system prompts"
        
        try:
            prompts_file = os.path.join(os.path.dirname(__file__), "system_prompts.json")
            
            # Load current prompts
            current_prompts = {}
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    current_prompts = json.load(f)
            
            if prompt_name not in current_prompts:
                return f"❌ Prompt '{prompt_name}' not found"
            
            # Remove the prompt
            del current_prompts[prompt_name]
            
            # Save back to file
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(current_prompts, f, indent=4, ensure_ascii=False)
            
            # Reload prompts
            self.system_prompts = load_system_prompts()
            
            return f"✅ Successfully deleted prompt '{prompt_name}'"
        except Exception as e:
            return f"❌ Error deleting prompt: {str(e)}"
    
    def load_prompt_for_editing(self, prompt_name: str) -> tuple:
        """Load a prompt for editing"""
        if not prompt_name or prompt_name not in self.system_prompts:
            return "", ""
        
        return prompt_name, self.system_prompts[prompt_name]
    
    def get_management_overview(self) -> str:
        """Get comprehensive management overview with modern HTML styling"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            # Get all available information with error handling
            health_result = self.client.test_connection()
        except Exception as e:
            health_result = {'error': str(e)}
            
        try:
            version_info = self.client.get_version_info()
        except Exception as e:
            version_info = {'success': False, 'error': str(e)}
            
        try:
            metrics_info = self.client.get_metrics()
        except Exception as e:
            metrics_info = {'success': False, 'error': str(e)}
        
        # Start HTML with header
        overview = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white;'>
            <h2 style='margin: 0 0 10px 0; font-size: 1.5em; font-weight: 600;'>🎛️ Model Serving Management</h2>
            <div style='font-size: 0.9em; opacity: 0.9;'>Last Updated: {timestamp}</div>
        </div>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'>
        """
        
        # Model Information Card
        models_data = health_result.get('models', {})
        model_status = "🟢 Online" if models_data.get('success') else "🔴 Offline"
        model_color = "#28a745" if models_data.get('success') else "#dc3545"
        available_models = ', '.join(models_data.get('available', [])) if models_data.get('available') else 'No models available'
        active_model = models_data.get('configured', 'Not configured')
        
        # Debug output
        print(f"DEBUG Models data: {models_data}")
        print(f"DEBUG Available models: {available_models}")
        print(f"DEBUG Active model: {active_model}")
        
        overview += f"""
        <div style='background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #007bff;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <div style='background: #007bff; color: white; border-radius: 8px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 1.2em;'>📊</div>
                <h3 style='margin: 0; color: #333; font-size: 1.1em;'>Model Information</h3>
            </div>
            <div style='color: #666; line-height: 1.6;'>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Status:</strong> <span style='color: {model_color}; font-weight: 600;'>{model_status}</span></div>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Available Models:</strong> <span style='color: #555;'>{available_models}</span></div>
                <div><strong style='color: #333;'>Active Model:</strong> <span style='color: #555;'>{active_model}</span></div>
            </div>
        </div>
        """
        
        # Server Health Card
        health_data = health_result.get('health', {})
        health_status = "✅ Healthy" if health_data.get('success') else "❌ Unhealthy"
        health_color = "#28a745" if health_data.get('success') else "#dc3545"
        response_time = f"{health_data.get('latency', 0)*1000:.0f}ms" if health_data.get('success') else "Connection failed"
        error_msg = health_data.get('error', 'Unknown error') if not health_data.get('success') else None
        
        # Debug output
        print(f"DEBUG Health data: {health_data}")
        print(f"DEBUG Health status: {health_status}")
        print(f"DEBUG Response time: {response_time}")
        
        health_card = f"""
        <div style='background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #28a745;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <div style='background: #28a745; color: white; border-radius: 8px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 1.2em;'>🏥</div>
                <h3 style='margin: 0; color: #333; font-size: 1.1em;'>Server Health</h3>
            </div>
            <div style='color: #666; line-height: 1.6;'>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Health Status:</strong> <span style='color: {health_color}; font-weight: 600;'>{health_status}</span></div>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Response Time:</strong> <span style='color: #555;'>{response_time}</span></div>"""
        
        if error_msg:
            health_card += f"""
                <div style='color: #dc3545; font-size: 0.9em; margin-top: 8px; padding: 8px; background: #f8d7da; border-radius: 4px;'><strong>Error:</strong> {error_msg}</div>"""
        
        health_card += """
            </div>
        </div>
        """
        overview += health_card
        
        # API Version Card
        version_display = "Unable to retrieve"
        version_details = []
        if version_info.get('success'):
            version_data = version_info.get('data', {})
            if isinstance(version_data, dict):
                raw_version = version_data.get('version', 'Version not reported')
                version_display = str(raw_version).replace('<', '&lt;').replace('>', '&gt;').strip()
                for key, value in version_data.items():
                    if key != 'version':
                        # Sanitize key and value to prevent HTML injection
                        safe_key = str(key).replace('<', '&lt;').replace('>', '&gt;').strip()
                        safe_value = str(value).replace('<', '&lt;').replace('>', '&gt;').strip()
                        version_details.append(f"<div style='margin-bottom: 4px;'><strong style='color: #333;'>{safe_key.title()}:</strong> <span style='color: #555;'>{safe_value}</span></div>")
            else:
                version_display = str(version_data)[:50] + "..." if len(str(version_data)) > 50 else str(version_data)
        else:
            version_error = version_info.get('error', 'Connection failed')
            safe_error = str(version_error).replace('<', '&lt;').replace('>', '&gt;').strip()
            version_details.append(f"<div style='color: #dc3545; font-size: 0.9em;'><strong>Error:</strong> {safe_error}</div>")
        
        # Debug output
        print(f"DEBUG Version info: {version_info}")
        print(f"DEBUG Version display: {version_display}")
        print(f"DEBUG Version details: {version_details}")
        
        # Sanitize version_details to prevent HTML injection
        import html
        safe_version_details = []
        for detail in version_details:
            # Ensure it's a string and clean
            detail_str = str(detail).strip()
            safe_version_details.append(detail_str)
        
        print(f"DEBUG Safe version details: {safe_version_details}")
        
        overview += f"""
        <div style='background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #6f42c1;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <div style='background: #6f42c1; color: white; border-radius: 8px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 1.2em;'>🔧</div>
                <h3 style='margin: 0; color: #333; font-size: 1.1em;'>API Version</h3>
            </div>
            <div style='color: #666; line-height: 1.6;'>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Version:</strong> <span style='color: #555;'>{str(version_display).strip()}</span></div>
                {''.join(safe_version_details) if safe_version_details else ''}
            </div>
        </div>
        """
        
        # Performance Summary Card
        chat_test = health_result.get('chat_test', {})
        streaming_test = health_result.get('streaming_test', {})
        
        chat_latency = f"{chat_test.get('latency', 0)*1000:.0f}ms" if chat_test.get('success') else "Connection failed"
        streaming_latency = f"{streaming_test.get('latency', 0)*1000:.0f}ms" if streaming_test.get('success') else "Not available"
        
        # Debug output
        print(f"DEBUG Chat test: {chat_test}")
        print(f"DEBUG Streaming test: {streaming_test}")
        print(f"DEBUG Chat latency: {chat_latency}")
        print(f"DEBUG Streaming latency: {streaming_latency}")
        
        overview += f"""
        <div style='background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #17a2b8;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <div style='background: #17a2b8; color: white; border-radius: 8px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 1.2em;'>📈</div>
                <h3 style='margin: 0; color: #333; font-size: 1.1em;'>Performance Summary</h3>
            </div>
            <div style='color: #666; line-height: 1.6;'>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Chat API Latency:</strong> <span style='color: #555;'>{chat_latency}</span></div>
                <div><strong style='color: #333;'>Streaming Latency:</strong> <span style='color: #555;'>{streaming_latency}</span></div>
            </div>
        </div>
        """
        
        # Configuration Card
        overview += f"""
        <div style='background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #fd7e14;'>
            <div style='display: flex; align-items: center; margin-bottom: 15px;'>
                <div style='background: #fd7e14; color: white; border-radius: 8px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-size: 1.2em;'>⚙️</div>
                <h3 style='margin: 0; color: #333; font-size: 1.1em;'>Configuration</h3>
            </div>
            <div style='color: #666; line-height: 1.6;'>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>API Endpoint:</strong> <code style='background: #f8f9fa; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; color: #495057;'>{self.config.api_endpoint}</code></div>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Default Temperature:</strong> <span style='color: #555;'>{self.config.default_temperature}</span></div>
                <div style='margin-bottom: 8px;'><strong style='color: #333;'>Default Max Tokens:</strong> <span style='color: #555;'>{self.config.default_max_tokens}</span></div>
                <div><strong style='color: #333;'>Auto-Stream Threshold:</strong> <span style='color: #555;'>{self.config.auto_stream_threshold} chars</span></div>
            </div>
        </div>
        """
        
        # Close grid container
        overview += "</div>"
        
        return overview
    
    def get_detailed_metrics(self) -> str:
        """Get detailed metrics in formatted view (fallback for text display)"""
        try:
            print("🔍 Getting detailed metrics...")
            metrics_result = self.client.get_metrics()
            
            if not metrics_result.get('success'):
                error_msg = f"## ❌ Metrics Unavailable\n\nError: {metrics_result.get('error', 'Unknown error')}"
                print(f"❌ Metrics API error: {metrics_result.get('error', 'Unknown error')}")
                return error_msg
            
            # Add new metrics data to collector
            raw_metrics = metrics_result.get('data', '')
            print(f"📊 Raw metrics received: {len(raw_metrics)} characters")
            
            if not raw_metrics:
                print("⚠️ No raw metrics data received")
                return "## ⚠️ No Metrics Data\n\nNo metrics data was returned from the server."
            
            self.metrics_collector.add_metrics_data(raw_metrics)
            print(f"📈 Metrics added to collector. Total tracked: {len(self.metrics_collector.metrics_data)}")
        except Exception as e:
            error_msg = f"## ❌ Error Loading Metrics\n\n{str(e)}"
            print(f"💥 Exception in get_detailed_metrics: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return error_msg
        
        # Return performance-focused dashboard
        try:
            if raw_metrics:
                print("🎨 Generating performance summary...")
                # Get the performance summary instead of raw metrics dump
                performance_summary = self.metrics_collector.format_performance_summary_text()
                print(f"✅ Performance summary generated: {len(performance_summary)} characters")
                
                # Also add a condensed raw metrics summary for reference
                parsed = self.metrics_collector.parse_prometheus_metrics(raw_metrics)
                if parsed:
                    # Filter out timestamp metrics for cleaner display
                    clean_parsed = {k: v for k, v in parsed.items() 
                                  if not (k.endswith('_created') or 'start_time_seconds' in k or 
                                         (v > 1.6e9 and any(hint in k.lower() for hint in ['time', 'created', 'start'])))}
                    
                    categories = defaultdict(list)
                    for metric_name in clean_parsed.keys():
                        category = self.metrics_collector.categorize_metric(metric_name)
                        categories[category].append(metric_name)
                    
                    # Add condensed category summary
                    performance_summary += "\n\n## 📋 **RAW METRICS SUMMARY**\n"
                    performance_summary += "=" * 40 + "\n"
                    
                    for category, metrics in sorted(categories.items()):
                        if metrics and category != 'other':  # Skip 'other' category for cleaner display
                            performance_summary += f"\n**{category.upper()}** ({len(metrics)} metrics)\n"
                            # Show top 3 most important metrics per category
                            for metric in sorted(metrics)[:3]:
                                value = clean_parsed.get(metric, 0)
                                formatted_value = self.metrics_collector.format_metric_value(metric, value)
                                performance_summary += f"   • {metric}: {formatted_value}\n"
                            if len(metrics) > 3:
                                performance_summary += f"   • ... and {len(metrics) - 3} more\n"
                    
                    # Show uncategorized metrics count if any
                    other_metrics = categories.get('other', [])
                    if other_metrics:
                        performance_summary += f"\n**UNCATEGORIZED** ({len(other_metrics)} metrics)"
                        if len(other_metrics) <= 3:
                            for metric in other_metrics:
                                value = clean_parsed.get(metric, 0)
                                formatted_value = self.metrics_collector.format_metric_value(metric, value)
                                performance_summary += f"\n   • {metric}: {formatted_value}"
            
                return performance_summary
            else:
                return "## ❌ **NO METRICS AVAILABLE**\n\nNo metrics data could be retrieved from the model server."
        except Exception as e:
            error_msg = f"## ❌ Error Generating Dashboard\n\n{str(e)}"
            print(f"💥 Exception in dashboard generation: {e}")
            import traceback
            print(f"🔍 Dashboard traceback: {traceback.format_exc()}")
            return error_msg
    

    def get_metrics_plots(self) -> Dict[str, Any]:
        """Get actionable performance dashboards that move the needle"""
        try:
            print("🚀 Creating actionable performance dashboards...")
            
            # Check if we have any data
            total_metrics = len(self.metrics_collector.metrics_data)
            total_timestamps = len(self.metrics_collector.timestamps)
            
            if total_metrics == 0:
                no_data_html = """
                <div style='text-align: center; padding: 60px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin: 20px; color: white;'>
                    <h2 style='margin: 0 0 20px 0; font-size: 2.5em;'>🚀</h2>
                    <h3 style='margin: 0 0 15px 0;'>Ready to Launch Performance Monitoring</h3>
                    <p style='margin: 0; opacity: 0.9; font-size: 1.1em;'>Start metrics collection to see real-time performance insights and optimization opportunities</p>
                    <div style='margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 8px;'>
                        <strong>What you'll get:</strong> Live performance alerts • Bottleneck detection • Optimization recommendations
                    </div>
                </div>
                """
                return {
                    'performance': no_data_html,
                    'health': no_data_html,
                    'efficiency': no_data_html,
                    'insights': no_data_html
                }
            
            # Get performance summary for dynamic content
            performance_summary = self.metrics_collector.get_performance_summary()
            
            # Create actionable dashboards
            dashboards = {}
            
            # 1. PERFORMANCE MONITORING - Most critical tab
            dashboards['performance'] = self.create_performance_monitoring_dashboard(performance_summary)
            
            # 2. HEALTH STATUS - System health alerts
            dashboards['health'] = self.create_health_status_dashboard(performance_summary)
            
            # 3. EFFICIENCY METRICS - Optimization opportunities  
            dashboards['efficiency'] = self.create_efficiency_dashboard(performance_summary)
            
            # 4. ACTIONABLE INSIGHTS - Recommendations and trends
            dashboards['insights'] = self.create_insights_dashboard(performance_summary)
            
            print(f"✅ Created 4 actionable dashboards with {total_metrics} metrics")
            return dashboards
            
        except Exception as e:
            error_msg = f"Error creating actionable dashboards: {str(e)}"
            print(f"💥 {error_msg}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            error_html = f"""
            <div style='text-align: center; padding: 40px; background: #fff5f5; border: 2px solid #fed7d7; border-radius: 12px; margin: 20px;'>
                <h3 style='color: #e53e3e; margin: 0 0 15px 0;'>⚠️ Dashboard Error</h3>
                <p style='color: #744210; margin: 0;'>{str(e)}</p>
            </div>
            """
            return {
                'performance': error_html,
                'health': error_html, 
                'efficiency': error_html,
                'insights': error_html
            }
    
    def create_performance_monitoring_dashboard(self, summary: Dict[str, Any]) -> str:
        """Create real-time performance monitoring dashboard"""
        if "error" in summary:
            return f"<div style='padding: 40px; text-align: center; color: #666;'>⚠️ {summary['error']}</div>"
        
        health = summary.get("model_health", {})
        perf = summary.get("performance", {})
        computed = summary.get("computed_metrics", {})
        
        # vLLM-specific performance indicators with platform team thresholds
        running = health.get("requests_running", 0)
        waiting = health.get("requests_waiting", 0) 
        swapped = health.get("requests_swapped", 0)
        cache_usage = health.get("gpu_cache_usage_pct", 0)
        rpm = computed.get("requests_per_minute", 0)
        ttft_avg = computed.get("avg_ttft_seconds", 0)
        success_rate = computed.get("success_rate_pct", 100)
        
        # vLLM Performance Scoring with industry-standard thresholds
        issues = []
        warnings = []
        
        # Critical thresholds (RED status)
        if cache_usage > 90:
            issues.append("GPU cache critically full")
        if waiting > 10:
            issues.append("Request queue backing up") 
        if ttft_avg > 5.0:
            issues.append("Slow time-to-first-token")
        if success_rate < 95:
            issues.append("High request failure rate")
        if swapped > 0:
            issues.append("Requests swapped to CPU")
            
        # Warning thresholds (YELLOW status)  
        if cache_usage > 75:
            warnings.append("GPU cache usage high")
        if waiting > 3:
            warnings.append("Requests queuing")
        if ttft_avg > 2.0:
            warnings.append("Elevated latency")
        if rpm < 5 and rpm > 0:
            warnings.append("Low throughput")
        if success_rate < 98:
            warnings.append("Some request failures")
        
        # Determine status
        if issues:
            status_color = "#ef4444"  # red
            status_text = "DEGRADED"
            status_icon = "🔴"
            status_details = " | ".join(issues[:2])  # Show top 2 issues
        elif warnings:
            status_color = "#f59e0b"  # amber
            status_text = "WARNING" 
            status_icon = "🟡"
            status_details = " | ".join(warnings[:2])  # Show top 2 warnings
        else:
            status_color = "#10b981"  # green
            status_text = "OPTIMAL"
            status_icon = "🟢" 
            status_details = "All systems performing within thresholds"
        
        return f"""
        <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; padding: 30px; margin: 20px; color: white;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;'>
                <h2 style='margin: 0; font-size: 1.8em; color: white;'>⚡ Real-Time Performance Monitor</h2>
                <div style='display: flex; align-items: center; background: {status_color}; padding: 12px 24px; border-radius: 25px; font-weight: bold;'>
                    <span style='font-size: 1.2em; margin-right: 8px;'>{status_icon}</span>
                    <span style='font-size: 1.1em;'>{status_text}</span>
                </div>
            </div>
            
            <!-- Status Details -->
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 15px; margin-bottom: 25px;'>
                <div style='font-size: 0.9em; color: #cbd5e1;'><strong>Status:</strong> {status_details}</div>
            </div>
            
            <!-- Key Performance Metrics -->
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px;'>
                <div style='background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.2em; margin-bottom: 8px;'>🎯</div>
                    <div style='font-size: 2em; font-weight: bold; color: #10b981;'>{running}</div>
                    <div style='font-size: 0.85em; opacity: 0.8;'>Active Requests</div>
                    <div style='font-size: 0.75em; color: #64748b; margin-top: 5px;'>Running on GPU</div>
                </div>
                
                <div style='background: rgba(251, 146, 60, 0.2); border: 2px solid #fb923c; border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.2em; margin-bottom: 8px;'>⏱️</div>
                    <div style='font-size: 2em; font-weight: bold; color: #fb923c;'>{waiting}</div>
                    <div style='font-size: 0.85em; opacity: 0.8;'>Queued Requests</div>
                    <div style='font-size: 0.75em; color: #64748b; margin-top: 5px;'>Waiting to process</div>
                </div>
                
                <div style='background: rgba(168, 85, 247, 0.2); border: 2px solid #a855f7; border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.2em; margin-bottom: 8px;'>💾</div>
                    <div style='font-size: 2em; font-weight: bold; color: #a855f7;'>{cache_usage:.1f}%</div>
                    <div style='font-size: 0.85em; opacity: 0.8;'>GPU Cache Usage</div>
                    <div style='font-size: 0.75em; color: #64748b; margin-top: 5px;'>KV Cache utilization</div>
                </div>
                
                <div style='background: rgba(34, 197, 94, 0.2); border: 2px solid #22c55e; border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.2em; margin-bottom: 8px;'>⚡</div>
                    <div style='font-size: 2em; font-weight: bold; color: #22c55e;'>{rpm:.1f}</div>
                    <div style='font-size: 0.85em; opacity: 0.8;'>Requests/min</div>
                    <div style='font-size: 0.75em; color: #64748b; margin-top: 5px;'>Throughput rate</div>
                </div>
                
                <!-- Additional vLLM-specific metrics -->
                <div style='background: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.2em; margin-bottom: 8px;'>🔄</div>
                    <div style='font-size: 2em; font-weight: bold; color: #ef4444;'>{swapped}</div>
                    <div style='font-size: 0.85em; opacity: 0.8;'>Swapped Requests</div>
                    <div style='font-size: 0.75em; color: #64748b; margin-top: 5px;'>Moved to CPU memory</div>
                </div>
                
                <div style='background: rgba(59, 130, 246, 0.2); border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.2em; margin-bottom: 8px;'>⏰</div>
                    <div style='font-size: 2em; font-weight: bold; color: #3b82f6;'>{ttft_avg*1000:.0f}ms</div>
                    <div style='font-size: 0.85em; opacity: 0.8;'>Avg TTFT</div>
                    <div style='font-size: 0.75em; color: #64748b; margin-top: 5px;'>Time to first token</div>
                </div>
            </div>
            
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: #f1f5f9;'>📊 vLLM Performance Metrics</h3>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'>
                    <div>
                        <div style='font-weight: bold; margin-bottom: 12px; color: #e2e8f0;'>🎯 Token Processing</div>
                        <div style='font-size: 1.1em; color: #10b981; margin-bottom: 6px;'>📝 {perf.get("total_prompt_tokens", 0):,} prompt tokens</div>
                        <div style='font-size: 1.1em; color: #3b82f6; margin-bottom: 6px;'>🎯 {perf.get("total_generation_tokens", 0):,} generated tokens</div>
                        <div style='font-size: 1.1em; color: #8b5cf6;'>🔢 {computed.get("total_tokens_processed", 0):,} total processed</div>
                    </div>
                    <div>
                        <div style='font-weight: bold; margin-bottom: 12px; color: #e2e8f0;'>⚡ Cache Performance</div>
                        <div style='font-size: 1.1em; color: #f59e0b; margin-bottom: 6px;'>🎯 {computed.get("cache_hit_rate_pct", 0):.1f}% hit rate</div>
                        <div style='font-size: 1.1em; color: #06b6d4; margin-bottom: 6px;'>💾 {cache_usage:.1f}% GPU usage</div>
                        <div style='font-size: 1.1em; color: #84cc16;'>🚀 {computed.get("cache_efficiency_score", 0):.0f}/100 efficiency</div>
                    </div>
                    <div>
                        <div style='font-weight: bold; margin-bottom: 12px; color: #e2e8f0;'>📈 Success Metrics</div>
                        <div style='font-size: 1.1em; color: #22c55e; margin-bottom: 6px;'>✅ {success_rate:.1f}% success rate</div>
                        <div style='font-size: 1.1em; color: #f97316; margin-bottom: 6px;'>🔄 {computed.get("preemption_rate", 0):.1f} preemptions/min</div>
                        <div style='font-size: 1.1em; color: #a855f7;'>⏱️ {ttft_avg:.3f}s avg TTFT</div>
                    </div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def create_health_status_dashboard(self, summary: Dict[str, Any]) -> str:
        """Create system health status dashboard with alerts"""
        if "error" in summary:
            return f"<div style='padding: 40px; text-align: center; color: #666;'>⚠️ {summary['error']}</div>"
        
        resources = summary.get("resource_usage", {})
        health = summary.get("model_health", {})
        stats = summary.get("request_stats", {})
        
        memory_gb = resources.get("memory_usage_gb", 0)
        cpu_time = resources.get("cpu_seconds_total", 0)
        open_fds = resources.get("open_file_descriptors", 0)
        preemptions = stats.get("preemptions", 0)
        
        # Health alerts
        alerts = []
        if memory_gb > 12:
            alerts.append(("🔴", "High Memory Usage", f"{memory_gb:.1f}GB - Consider scaling"))
        if preemptions > 5:
            alerts.append(("🟡", "Request Preemptions", f"{preemptions} preempted requests detected"))
        if open_fds > 500:
            alerts.append(("🟡", "High File Descriptor Usage", f"{open_fds} open FDs"))
        
        if not alerts:
            alerts.append(("🟢", "System Healthy", "All systems operating normally"))
        
        return f"""
        <div style='background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-radius: 16px; padding: 30px; margin: 20px; color: white;'>
            <h2 style='margin: 0 0 30px 0; font-size: 1.8em; color: white;'>🏥 System Health Monitor</h2>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;'>
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px;'>
                    <h3 style='margin: 0 0 15px 0; color: #d1fae5;'>💾 Memory Health</h3>
                    <div style='font-size: 2em; font-weight: bold; color: {"#ef4444" if memory_gb > 12 else "#10b981"};'>{memory_gb:.2f} GB</div>
                    <div style='margin-top: 10px; font-size: 0.9em; opacity: 0.8;'>
                        Status: {"⚠️ High Usage" if memory_gb > 12 else "✅ Normal"}
                    </div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px;'>
                    <h3 style='margin: 0 0 15px 0; color: #d1fae5;'>⚙️ CPU Performance</h3>
                    <div style='font-size: 2em; font-weight: bold; color: #3b82f6;'>{cpu_time:.1f}s</div>
                    <div style='margin-top: 10px; font-size: 0.9em; opacity: 0.8;'>
                        Total CPU Time Used
                    </div>
                </div>
            </div>
            
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: #f1f5f9;'>🚨 Active Alerts</h3>
                <div style='display: flex; flex-direction: column; gap: 12px;'>
                    {"".join([f'''
                    <div style='display: flex; align-items: center; padding: 12px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;'>
                        <span style='font-size: 1.5em; margin-right: 15px;'>{icon}</span>
                        <div>
                            <div style='font-weight: bold;'>{title}</div>
                            <div style='font-size: 0.9em; opacity: 0.8;'>{desc}</div>
                        </div>
                    </div>
                    ''' for icon, title, desc in alerts])}
                </div>
            </div>
            
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: #f1f5f9;'>📊 System Metrics</h3>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
                    <div>
                        <div style='font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;'>Successful Requests</div>
                        <div style='font-size: 1.5em; font-weight: bold; color: #10b981;'>{stats.get("successful_requests", 0):,}</div>
                    </div>
                    <div>
                        <div style='font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;'>Request Preemptions</div>
                        <div style='font-size: 1.5em; font-weight: bold; color: {"#ef4444" if preemptions > 5 else "#10b981"};'>{preemptions:,}</div>
                    </div>
                    <div>
                        <div style='font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;'>Open File Descriptors</div>
                        <div style='font-size: 1.5em; font-weight: bold; color: {"#f59e0b" if open_fds > 500 else "#10b981"};'>{open_fds:,}</div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def create_efficiency_dashboard(self, summary: Dict[str, Any]) -> str:
        """Create efficiency optimization dashboard"""
        if "error" in summary:
            return f"<div style='padding: 40px; text-align: center; color: #666;'>⚠️ {summary['error']}</div>"
        
        computed = summary.get("computed_metrics", {})
        health = summary.get("model_health", {})
        
        cache_hit_rate = computed.get("cache_hit_rate_pct", 0)
        cache_usage = health.get("gpu_cache_usage_pct", 0)
        total_tokens = computed.get("total_tokens_processed", 0)
        rpm = computed.get("requests_per_minute", 0)
        
        # Efficiency scoring
        efficiency_score = 0
        if cache_hit_rate > 80: efficiency_score += 25
        elif cache_hit_rate > 60: efficiency_score += 15
        elif cache_hit_rate > 40: efficiency_score += 10
        
        if cache_usage < 70: efficiency_score += 25
        elif cache_usage < 85: efficiency_score += 15
        elif cache_usage < 95: efficiency_score += 10
        
        if rpm > 5: efficiency_score += 25
        elif rpm > 2: efficiency_score += 15
        elif rpm > 1: efficiency_score += 10
        
        if total_tokens > 10000: efficiency_score += 25
        elif total_tokens > 5000: efficiency_score += 15
        elif total_tokens > 1000: efficiency_score += 10
        
        # Optimization recommendations
        recommendations = []
        if cache_hit_rate < 60:
            recommendations.append(("🎯", "Improve Cache Hit Rate", "Consider warming up the cache with common queries"))
        if cache_usage > 85:
            recommendations.append(("💾", "Cache Usage High", "Monitor for potential OOM issues, consider batch size tuning"))
        if rpm < 2:
            recommendations.append(("⚡", "Low Request Rate", "Consider optimizing model parameters or increasing concurrency"))
        
        if not recommendations:
            recommendations.append(("✅", "Well Optimized", "System is performing efficiently"))
        
        return f"""
        <div style='background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); border-radius: 16px; padding: 30px; margin: 20px; color: white;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;'>
                <h2 style='margin: 0; font-size: 1.8em; color: white;'>🚀 Efficiency Optimizer</h2>
                <div style='background: rgba(255, 255, 255, 0.2); padding: 15px 25px; border-radius: 25px; text-align: center;'>
                    <div style='font-size: 2em; font-weight: bold; color: {"#10b981" if efficiency_score > 70 else "#f59e0b" if efficiency_score > 40 else "#ef4444"};'>{efficiency_score}%</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Efficiency Score</div>
                </div>
            </div>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;'>
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2em; margin-bottom: 10px;'>🎯</div>
                    <div style='font-size: 2em; font-weight: bold; color: {"#10b981" if cache_hit_rate > 70 else "#f59e0b" if cache_hit_rate > 50 else "#ef4444"};'>{cache_hit_rate:.1f}%</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Cache Hit Rate</div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2em; margin-bottom: 10px;'>💾</div>
                    <div style='font-size: 2em; font-weight: bold; color: {"#ef4444" if cache_usage > 85 else "#f59e0b" if cache_usage > 70 else "#10b981"};'>{cache_usage:.1f}%</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Cache Utilization</div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2em; margin-bottom: 10px;'>⚡</div>
                    <div style='font-size: 2em; font-weight: bold; color: {"#10b981" if rpm > 3 else "#f59e0b" if rpm > 1 else "#ef4444"};'>{rpm:.1f}</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Throughput (req/min)</div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2em; margin-bottom: 10px;'>🔢</div>
                    <div style='font-size: 2em; font-weight: bold; color: #3b82f6;'>{total_tokens:,}</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Total Tokens</div>
                </div>
            </div>
            
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: #f1f5f9;'>💡 Optimization Recommendations</h3>
                <div style='display: flex; flex-direction: column; gap: 12px;'>
                    {"".join([f'''
                    <div style='display: flex; align-items: center; padding: 12px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;'>
                        <span style='font-size: 1.5em; margin-right: 15px;'>{icon}</span>
                        <div>
                            <div style='font-weight: bold;'>{title}</div>
                            <div style='font-size: 0.9em; opacity: 0.8;'>{desc}</div>
                        </div>
                    </div>
                    ''' for icon, title, desc in recommendations])}
                </div>
            </div>
        </div>
        """
    
    def create_insights_dashboard(self, summary: Dict[str, Any]) -> str:
        """Create actionable insights and trends dashboard"""
        if "error" in summary:
            return f"<div style='padding: 40px; text-align: center; color: #666;'>⚠️ {summary['error']}</div>"
        
        # Get some trend data
        with self.metrics_collector.lock:
            timestamps = list(self.metrics_collector.timestamps)
            running_requests = self.metrics_collector.metrics_data.get('vllm:num_requests_running', [])
            
        time_range = "No data"
        if len(timestamps) >= 2:
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
            if duration > 3600:
                time_range = f"{duration/3600:.1f} hours"
            elif duration > 60:
                time_range = f"{duration/60:.1f} minutes"
            else:
                time_range = f"{duration:.0f} seconds"
        
        # Generate insights
        insights = [
            ("📊", "Data Collection", f"Monitoring active for {time_range} with {len(timestamps)} data points"),
            ("🎯", "Performance Trend", "Token processing rate appears stable" if len(running_requests) > 0 else "Awaiting more data for trend analysis"),
            ("💡", "Optimization Opportunity", "Consider enabling request batching for improved throughput"),
            ("🔍", "Monitoring Health", f"Collecting {len(self.metrics_collector.metrics_data)} unique metrics successfully")
        ]
        
        return f"""
        <div style='background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%); border-radius: 16px; padding: 30px; margin: 20px; color: white;'>
            <h2 style='margin: 0 0 30px 0; font-size: 1.8em; color: white;'>🧠 Actionable Insights</h2>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;'>
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>⏱️</div>
                    <div style='font-size: 1.5em; font-weight: bold; color: #67e8f9;'>{time_range}</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Monitoring Duration</div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>📈</div>
                    <div style='font-size: 1.5em; font-weight: bold; color: #34d399;'>{len(timestamps)}</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Data Points Collected</div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>🎯</div>
                    <div style='font-size: 1.5em; font-weight: bold; color: #fbbf24;'>{len(self.metrics_collector.metrics_data)}</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>Active Metrics</div>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 20px; text-align: center;'>
                    <div style='font-size: 2.5em; margin-bottom: 10px;'>🚀</div>
                    <div style='font-size: 1.5em; font-weight: bold; color: #a78bfa;'>{"ACTIVE" if len(running_requests) > 0 else "READY"}</div>
                    <div style='font-size: 0.9em; opacity: 0.8;'>System Status</div>
                </div>
            </div>
            
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: #f1f5f9;'>🔍 Performance Insights</h3>
                <div style='display: flex; flex-direction: column; gap: 12px;'>
                    {"".join([f'''
                    <div style='display: flex; align-items: center; padding: 12px; background: rgba(255, 255, 255, 0.1); border-radius: 8px;'>
                        <span style='font-size: 1.5em; margin-right: 15px;'>{icon}</span>
                        <div>
                            <div style='font-weight: bold;'>{title}</div>
                            <div style='font-size: 0.9em; opacity: 0.8;'>{desc}</div>
                        </div>
                    </div>
                    ''' for icon, title, desc in insights])}
                </div>
            </div>
            
            <div style='background: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: #f1f5f9;'>🎯 Next Steps</h3>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;'>
                    <div style='padding: 15px; background: rgba(16, 185, 129, 0.2); border-radius: 8px; border-left: 4px solid #10b981;'>
                        <div style='font-weight: bold; color: #10b981; margin-bottom: 5px;'>📊 Continue Monitoring</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Let the system collect more data for better trend analysis</div>
                    </div>
                    <div style='padding: 15px; background: rgba(59, 130, 246, 0.2); border-radius: 8px; border-left: 4px solid #3b82f6;'>
                        <div style='font-weight: bold; color: #3b82f6; margin-bottom: 5px;'>⚡ Test Load</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Run some chat requests to see performance under load</div>
                    </div>
                    <div style='padding: 15px; background: rgba(168, 85, 247, 0.2); border-radius: 8px; border-left: 4px solid #a855f7;'>
                        <div style='font-weight: bold; color: #a855f7; margin-bottom: 5px;'>🔧 Tune Parameters</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Adjust temperature and token limits based on usage patterns</div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def get_api_capabilities(self) -> str:
        """Get API capabilities from OpenAPI spec"""
        spec_result = self.client.get_openapi_spec()
        
        if not spec_result.get('success'):
            return f"## ❌ API Specification Unavailable\n\nError: {spec_result.get('error', 'Unknown error')}"
        
        spec = spec_result.get('data', {})
        capabilities = "## 🚀 API Capabilities\n\n"
        
        # Extract basic info
        info = spec.get('info', {})
        if info:
            capabilities += f"**API Title:** {info.get('title', 'Unknown')}\n"
            capabilities += f"**API Version:** {info.get('version', 'Unknown')}\n\n"
        
        # Extract available endpoints
        paths = spec.get('paths', {})
        if paths:
            capabilities += "### Available Endpoints:\n"
            
            # Group endpoints by category
            categorized = {
                'Core': [],
                'Models': [],
                'Health': [],
                'Utilities': [],
                'Other': []
            }
            
            for path, methods in paths.items():
                if '/v1/' in path:
                    categorized['Core'].append(path)
                elif 'model' in path.lower():
                    categorized['Models'].append(path)
                elif any(x in path.lower() for x in ['health', 'ping', 'metrics']):
                    categorized['Health'].append(path)
                elif any(x in path.lower() for x in ['tokenize', 'version', 'docs']):
                    categorized['Utilities'].append(path)
                else:
                    categorized['Other'].append(path)
            
            for category, endpoints in categorized.items():
                if endpoints:
                    capabilities += f"\n**{category}:**\n"
                    for endpoint in sorted(endpoints):
                        capabilities += f"- `{endpoint}`\n"
        
        return capabilities
    
    def run_diagnostics(self) -> str:
        """Run comprehensive diagnostics"""
        print("🔍 Running enhanced diagnostics...")
        results = self.client.test_connection()
        
        diagnostic_text = f"# 🔍 Enhanced Connection Diagnostics\n"
        diagnostic_text += f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Health check
        health = results.get('health', {})
        status = "✅ PASS" if health.get('success') else "❌ FAIL"
        diagnostic_text += f"## 1. Health Check: {status}\n"
        if health.get('success'):
            diagnostic_text += f"- Latency: {health.get('latency', 0):.2f}s\n"
        else:
            diagnostic_text += f"- Error: {health.get('error', 'Unknown')}\n"
        diagnostic_text += "\n"
        
        # Models check
        models = results.get('models', {})
        status = "✅ PASS" if models.get('success') else "❌ FAIL"
        diagnostic_text += f"## 2. Models Check: {status}\n"
        if models.get('success'):
            available = models.get('available', [])
            configured = models.get('configured', '')
            match = models.get('match', False)
            diagnostic_text += f"- Available: {available}\n"
            diagnostic_text += f"- Configured: {configured}\n"
            diagnostic_text += f"- Match: {'✅' if match else '❌'}\n"
            if not match and available:
                diagnostic_text += f"- **💡 Try:** {available[0]}\n"
        else:
            diagnostic_text += f"- Error: {models.get('error', 'Unknown')}\n"
        diagnostic_text += "\n"
        
        # Chat test
        chat_test = results.get('chat_test', {})
        status = "✅ PASS" if chat_test.get('success') else "❌ FAIL"
        diagnostic_text += f"## 3. Chat API Test: {status}\n"
        if chat_test.get('success'):
            diagnostic_text += f"- Latency: {chat_test.get('latency', 0):.2f}s\n"
        else:
            diagnostic_text += f"- Status: {chat_test.get('status', 'Unknown')}\n"
            diagnostic_text += f"- Error: {chat_test.get('error', chat_test.get('response', 'Unknown'))}\n"
        diagnostic_text += "\n"
        
        # Streaming test
        streaming_test = results.get('streaming_test', {})
        status = "✅ PASS" if streaming_test.get('success') else "❌ FAIL"
        diagnostic_text += f"## 4. Streaming Test: {status}\n"
        if streaming_test.get('success'):
            diagnostic_text += f"- Latency: {streaming_test.get('latency', 0):.2f}s\n"
            diagnostic_text += f"- Content received: {'✅' if streaming_test.get('content_received') else '❌'}\n"
        else:
            diagnostic_text += f"- Error: {streaming_test.get('error', 'Unknown')}\n"
        diagnostic_text += "\n"
        
        # Recommendations
        diagnostic_text += "## 💡 Recommendations\n"
        if not health.get('success'):
            diagnostic_text += "- Check network connectivity and API endpoint\n"
        if not models.get('match', True):
            diagnostic_text += "- Update model name in configuration\n"
        if not chat_test.get('success'):
            diagnostic_text += "- API may be overloaded, try again later\n"
        if not streaming_test.get('success'):
            diagnostic_text += "- Streaming disabled, large contexts may timeout\n"
        
        return diagnostic_text
    
    def create_interface(self) -> gr.Blocks:
        """Create enhanced Gradio interface"""
        
        # Custom CSS for better layout and readability
        custom_css = """
        /* Reset default Gradio constraints */
        main, .main, .w-full {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        /* Make the interface use full width */
        .container {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
        }
        
        /* Full width for main gradio container */
        .gradio-container {
            max-width: 100% !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 20px !important;
        }
        
        /* Override any max-width constraints */
        div[class*="max-w-"] {
            max-width: 100% !important;
        }
        
        /* Increase chat message font size and spacing */
        .message-wrap {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        
        /* Better spacing for chat bubbles */
        .message {
            padding: 12px 20px !important;
            margin: 10px 0 !important;
        }
        
        /* Full width chat area */
        #chatbot {
            max-width: none !important;
            width: 100% !important;
        }
        
        /* Better code block styling */
        .message pre {
            background-color: #f4f4f4 !important;
            color: #333333 !important;
            padding: 12px !important;
            border-radius: 6px !important;
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        
        /* Ensure code within pre blocks is visible */
        .message pre code {
            color: #333333 !important;
            background-color: transparent !important;
        }
        
        /* Style inline code */
        .message code {
            background-color: #f4f4f4 !important;
            color: #333333 !important;
            padding: 2px 6px !important;
            border-radius: 3px !important;
            font-size: 0.9em !important;
        }
        
        /* Improve button spacing */
        .gr-button {
            margin: 2px !important;
        }
        
        /* Configuration panel styling */
        .config-panel {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            height: 100%;
            min-width: 280px;
        }
        
        /* Tab container full width */
        .tabs {
            width: 100% !important;
        }
        
        /* Chat tab content full width */
        .tabitem {
            width: 100% !important;
        }
        
        /* Message input full width */
        .gr-text-input {
            width: 100% !important;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .config-panel {
                min-width: 100%;
            }
        }
        
        /* Hide duplicate progress bars */
        .progress-bar:nth-of-type(n+2) {
            display: none !important;
        }
        
        /* Metrics tab styling */
        .metrics-tab {
            padding: 10px;
        }
        
        /* Plot containers */
        .plot-container {
            margin: 10px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Metrics controls styling */
        .metrics-controls {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        /* Color coding for metric categories */
        .memory-metrics { border-left: 4px solid #FF6B6B; }
        .transaction-metrics { border-left: 4px solid #4ECDC4; }
        .token-metrics { border-left: 4px solid #45B7D1; }
        .model-metrics { border-left: 4px solid #96CEB4; }
        
        /* Limit progress bars to single instance */
        .gradio-container .wrap:has(.progress-bar) .progress-bar ~ .progress-bar {
            display: none !important;
        }
        
        /* Clean up processing indicators */
        .processing-indicator {
            max-height: 4px !important;
        }
        """
        
        with gr.Blocks(title="Telco-AIX SME Web Interface", theme=gr.themes.Soft(), css=custom_css) as interface:
            
            # Header - more compact
            with gr.Row():
                with gr.Column(scale=4):
                    # Get active model from API if possible
                    try:
                        test_result = self.client.test_connection()
                        active_model = test_result.get('models', {}).get('configured', self.config.model_name)
                    except:
                        active_model = self.config.model_name
                    
                    gr.Markdown(
                        f"""
                        # 🤖 Telco-AIX SME Web Interface
                        **Connected to:** {active_model} | **Features:** Smart streaming, timeout handling, context optimization
                        """
                    )
                with gr.Column(scale=1):
                    # Status indicator in header
                    if self.client.health_check():
                        gr.Markdown("✅ **Status:** Online")
                    else:
                        gr.Markdown("❌ **Status:** Offline")
            
            # Session management row
            with gr.Row():
                with gr.Column(scale=2):
                    session_id_input = gr.Textbox(
                        label="🔑 Session ID",
                        placeholder="Leave empty for new session or enter existing session ID",
                        value="",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    load_session_btn = gr.Button("📂 Load Session", variant="secondary")
                    new_session_btn = gr.Button("🆕 New Session", variant="primary")
            
            # Sessions list display (collapsible)
            with gr.Accordion("📂 Active Sessions", open=False) as sessions_accordion:
                with gr.Row():
                    with gr.Column(scale=3):
                        sessions_display = gr.Markdown("Click 'List Sessions' to view active sessions...")
                    with gr.Column(scale=1):
                        gr.Markdown("**Quick Load:**")
                        session_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Session",
                            interactive=True,
                            show_label=False
                        )
                        quick_load_btn = gr.Button("⚡ Quick Load", variant="primary", size="sm")
            
            # Main content area with tabs for better organization
            with gr.Tabs():
                with gr.TabItem("💬 Chat"):
                    with gr.Row():
                        # Main chat column - use most of the screen
                        with gr.Column(scale=5):
                            # Chat interface
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=700,
                                elem_id="chatbot",
                                type="messages",
                                show_label=False,
                                container=True,
                                scale=1,
                                layout="panel"
                            )
                            
                            # Message input area
                            with gr.Row():
                                msg = gr.Textbox(
                                    label="Message",
                                    placeholder="Type your message here... (Auto-streaming for large contexts)",
                                    lines=3,
                                    scale=5,
                                    show_label=False
                                )
                            
                            # Action buttons
                            with gr.Row():
                                submit = gr.Button("Send", variant="primary", scale=1)
                                clear = gr.Button("🗑️ Clear", scale=1)
                                export = gr.Button("📥 Export", scale=1)
                                file_upload = gr.File(
                                    label="📎 Attach",
                                    file_types=[".txt", ".md", ".csv", ".json", ".py"],
                                    file_count="single",
                                    scale=1
                                )
                            
                            # Context info
                            context_info = gr.Markdown(f"**Context:** Ready | **Mode:** Direct | **Temp:** {self.config.default_temperature} | **Tokens:** {self.config.default_max_tokens}", elem_id="context-info")
                            
                            # Export output (hidden by default)
                            export_output = gr.Textbox(
                                label="📄 Exported Conversation",
                                visible=False,
                                lines=10
                            )
                        
                        # Configuration sidebar - narrower
                        with gr.Column(scale=1, elem_classes="config-panel"):
                            gr.HTML("<h3 style='color: #333; margin: 10px 0;'>⚙️ Settings</h3>")
                            
                            # System Prompt Section
                            with gr.Group():
                                system_dropdown = gr.Dropdown(
                                    choices=list(self.system_prompts.keys()),
                                    value="Telco Expert",
                                    label="System Prompt",
                                    scale=1
                                )
                            
                            # Model Parameters Section  
                            gr.HTML("<h4 style='color: #555; margin: 15px 0 8px 0;'>🎛️ Model Parameters</h4>")
                            with gr.Group():
                                temperature = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=self.config.default_temperature,
                                    step=0.1,
                                    label="🌡️ Temperature",
                                    scale=1,
                                    info="Controls randomness (0=focused, 1=creative)",
                                    interactive=True
                                )
                                
                                max_tokens = gr.Slider(
                                    minimum=100,
                                    maximum=8192,
                                    value=self.config.default_max_tokens,
                                    step=100,
                                    label="📏 Max Tokens",
                                    scale=1,
                                    info="Maximum response length",
                                    interactive=True
                                )
                                
                                # Parameter status display
                                param_status = gr.Markdown(
                                    f"**Current:** Temp={self.config.default_temperature} | Tokens={self.config.default_max_tokens}",
                                    elem_id="param-status"
                                )
                            
                            # Custom Prompt Section - stretched to fill remaining space
                            gr.HTML("<h4 style='color: #555; margin: 15px 0 8px 0;'>📝 Prompt Override</h4>")
                            with gr.Group():
                                custom_system = gr.Textbox(
                                    label="🎯 Selected System Prompt Detail",
                                    placeholder="Override selected template with custom prompt...",
                                    lines=12,
                                    max_lines=20,
                                    scale=1,
                                    show_copy_button=True,
                                    info="To edit templates permanently, use the 📝 Prompt Manager tab.",
                                    elem_classes="system-prompt-detail"
                                )
                
                # Additional tabs for better organization
                with gr.TabItem("🎛️ Management"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Overview section with auto-refresh
                            gr.Markdown("### 📊 Model Serving Overview")
                            with gr.Row():
                                refresh_overview_btn = gr.Button("🔄 Refresh Overview", variant="primary")
                                auto_refresh_checkbox = gr.Checkbox(label="Auto-refresh (30s)", value=False)
                            
                            management_overview = gr.Markdown(
                                value="Click 'Refresh Overview' to load current status...",
                                elem_id="management-overview"
                            )
                            
                        with gr.Column(scale=1):
                            gr.Markdown("### 🛠️ Quick Actions")
                            diagnostics_btn = gr.Button("🔍 Run Full Diagnostics", variant="secondary", size="sm")
                            test_streaming_btn = gr.Button("🧪 Test Streaming", variant="secondary", size="sm")
                            test_ui_btn = gr.Button("🎯 Test UI Update", variant="secondary", size="sm")
                            
                            gr.Markdown("### 📝 Status")
                            last_update_display = gr.Textbox(
                                label="Last Update",
                                value="Not updated yet",
                                interactive=False,
                                max_lines=1
                            )
                    
                    with gr.Row():
                        # Tabbed sections for detailed information
                        with gr.Tabs():
                            with gr.TabItem("📊 Metrics"):
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        # Metrics controls
                                        with gr.Row():
                                            start_collection_btn = gr.Button("▶️ Start Collection", variant="primary", size="sm")
                                            stop_collection_btn = gr.Button("⏸️ Stop Collection", variant="secondary", size="sm")
                                            debug_btn = gr.Button("🔧 Debug Info", variant="secondary", size="sm")
                                            refresh_plots_btn = gr.Button("🔄 Refresh Plots", variant="primary", size="sm")
                                            
                                        with gr.Row():
                                            pull_interval_slider = gr.Slider(
                                                minimum=5,
                                                maximum=300,
                                                value=30,
                                                step=5,
                                                label="Pull Interval (seconds)",
                                                interactive=True
                                            )
                                        
                                        collection_status = gr.Textbox(
                                            label="Collection Status",
                                            value="Not started",
                                            interactive=False,
                                            max_lines=1
                                        )
                                
                                # Consolidated vLLM metrics dashboard - 4 core categories
                                with gr.Row():
                                    with gr.Tabs():
                                        with gr.TabItem("⚡ Performance Monitor", elem_classes="metrics-tab"):
                                            performance_dashboard = gr.HTML(
                                                label="Real-Time Performance Monitoring",
                                                elem_id="performance-dashboard",
                                                value="<div style='text-align: center; padding: 40px; color: #666;'>Click 'Refresh Plots' to load performance monitor</div>"
                                            )
                                        
                                        with gr.TabItem("🏥 Health Status", elem_classes="metrics-tab"):
                                            health_dashboard = gr.HTML(
                                                label="System Health & Alerts",
                                                elem_id="health-dashboard",
                                                value="<div style='text-align: center; padding: 40px; color: #666;'>Click 'Refresh Plots' to load health status</div>"
                                            )
                                        
                                        with gr.TabItem("🚀 Efficiency", elem_classes="metrics-tab"):
                                            efficiency_dashboard = gr.HTML(
                                                label="Performance Optimization & Efficiency Scoring",
                                                elem_id="efficiency-dashboard",
                                                value="<div style='text-align: center; padding: 40px; color: #666;'>Click 'Refresh Plots' to load efficiency metrics</div>"
                                            )
                                        
                                        with gr.TabItem("🧠 Insights", elem_classes="metrics-tab"):
                                            insights_dashboard = gr.HTML(
                                                label="Actionable Insights & Recommendations",
                                                elem_id="insights-dashboard",
                                                value="<div style='text-align: center; padding: 40px; color: #666;'>Click 'Refresh Plots' to load insights</div>"
                                            )
                                
                                # Archive management section
                                with gr.Accordion("💾 Data Archive", open=False):
                                    with gr.Row():
                                        export_btn = gr.Button("📤 Export Data", variant="secondary", size="sm")
                                        import_btn = gr.Button("📥 Import Data", variant="secondary", size="sm")
                                        clear_archive_btn = gr.Button("🗑️ Clear Archive", variant="stop", size="sm")
                                    
                                    with gr.Row():
                                        export_filename = gr.Textbox(
                                            label="Export Filename (optional)",
                                            placeholder="metrics_export_custom.json",
                                            scale=2
                                        )
                                        import_file_dropdown = gr.Dropdown(
                                            label="Import File",
                                            choices=[],
                                            scale=2,
                                            interactive=True
                                        )
                                        refresh_files_btn = gr.Button("🔄", size="sm", scale=1)
                                    
                                    archive_status = gr.Textbox(
                                        label="Archive Status",
                                        value="Archive operations will show status here...",
                                        interactive=False,
                                        lines=2
                                    )
                                
                                # Fallback text display
                                with gr.Accordion("📋 Text Summary", open=False):
                                    metrics_output = gr.Markdown(
                                        value="Metrics summary will appear here after collection starts...",
                                        elem_id="metrics-display"
                                    )
                            
                            with gr.TabItem("🚀 API Capabilities"):
                                with gr.Row():
                                    refresh_capabilities_btn = gr.Button("🔄 Refresh API Capabilities", variant="primary", size="sm")
                                capabilities_output = gr.Markdown(
                                    value="API capabilities will appear here after refresh...",
                                    elem_id="capabilities-display"
                                )
                            
                            with gr.TabItem("🔍 Diagnostics Log"):
                                with gr.Row():
                                    refresh_diagnostics_btn = gr.Button("🔄 Refresh Diagnostics", variant="primary", size="sm")
                                    run_diagnostics_btn = gr.Button("🔍 Run Full Diagnostics", variant="secondary", size="sm")
                                diagnostics_output = gr.Textbox(
                                    label="📊 Diagnostics Report",
                                    lines=15,
                                    max_lines=25,
                                    value="Click 'Refresh Diagnostics' or 'Run Full Diagnostics' to test all endpoints..."
                                )
                
                with gr.TabItem("📂 Sessions"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Session Information")
                            sessions_info = gr.Textbox(
                                label="📋 Active Sessions",
                                lines=15,
                                max_lines=20,
                                value="Click 'List Sessions' to view active sessions...",
                                interactive=False
                            )
                            
                            with gr.Row():
                                list_sessions_btn = gr.Button("📋 List Sessions", variant="primary")
                                cleanup_sessions_btn = gr.Button("🧹 Cleanup Expired", variant="secondary")
                
                with gr.TabItem("📝 Prompt Manager"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### System Prompt Management")
                            
                            # Existing prompts section
                            gr.Markdown("### 📚 Existing Prompts")
                            existing_prompt_dropdown = gr.Dropdown(
                                choices=list(self.system_prompts.keys()),
                                value=list(self.system_prompts.keys())[0] if self.system_prompts else None,
                                label="Select Prompt to Edit",
                                interactive=True
                            )
                            
                            load_prompt_btn = gr.Button("📖 Load Selected Prompt", variant="secondary")
                            
                            with gr.Row():
                                reload_prompts_btn = gr.Button("🔄 Reload from File", variant="secondary")
                                delete_prompt_btn = gr.Button("🗑️ Delete Selected", variant="stop")
                            
                            prompt_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Ready to manage prompts..."
                            )
                            
                        with gr.Column(scale=2):
                            gr.Markdown("### ✏️ Edit Prompt")
                            edit_prompt_name = gr.Textbox(
                                label="Prompt Name",
                                placeholder="e.g., Security Expert, Project Manager...",
                                interactive=True
                            )
                            edit_prompt_content = gr.Textbox(
                                label="Prompt Content",
                                placeholder="Enter the full system prompt...",
                                lines=10,
                                max_lines=20,
                                interactive=True
                            )
                            
                            with gr.Row():
                                save_prompt_btn = gr.Button("💾 Save/Update Prompt", variant="primary")
                                clear_form_btn = gr.Button("🆕 Clear Form", variant="secondary")
                            
                            gr.Markdown(
                                """
                                ### 💡 Usage Tips
                                - **Load**: Select a prompt and click "Load" to edit
                                - **Save**: Updates existing or creates new prompt
                                - **Delete**: Removes selected prompt permanently
                                - **Clear**: Start fresh with empty form
                                - All changes are saved to `system_prompts.json`
                                """
                            )
            
            # Event handlers
            def update_system_prompt(selection):
                return self.system_prompts.get(selection, "")
            
            def update_context_info(history, message, temp, tokens):
                if not history:
                    return f"**Context:** Ready | **Mode:** Direct | **Temp:** {temp} | **Tokens:** {tokens}"
                
                # Calculate total characters with support for both formats
                total_chars = 0
                for h in history:
                    if isinstance(h, dict) and 'content' in h:
                        total_chars += len(h['content'])
                    elif isinstance(h, (list, tuple)) and len(h) == 2:
                        total_chars += len(h[0]) + len(h[1])
                total_chars += len(message or "")
                mode = "🌊 Streaming" if total_chars > 4000 else "⚡ Direct"
                return f"**Context:** {total_chars} chars | **Mode:** {mode} | **Temp:** {temp} | **Tokens:** {tokens}"
            
            def update_param_status(temp, tokens):
                return f"**Current:** Temp={temp} | Tokens={tokens}"
            
            # Wire up the system dropdown
            system_dropdown.change(
                update_system_prompt,
                inputs=[system_dropdown],
                outputs=[custom_system]
            )
            
            # Update context info on message change
            msg.change(
                update_context_info,
                inputs=[chatbot, msg, temperature, max_tokens],
                outputs=[context_info]
            )
            
            # Update context info and param status when sliders change
            temperature.change(
                lambda h, m, t, tok: [update_context_info(h, m, t, tok), update_param_status(t, tok)],
                inputs=[chatbot, msg, temperature, max_tokens],
                outputs=[context_info, param_status]
            )
            
            max_tokens.change(
                lambda h, m, t, tok: [update_context_info(h, m, t, tok), update_param_status(t, tok)],
                inputs=[chatbot, msg, temperature, max_tokens],
                outputs=[context_info, param_status]
            )
            
            # Management tab handlers
            def refresh_management_overview():
                """Refresh all management information"""
                try:
                    overview = self.get_management_overview()
                except Exception as e:
                    overview = f"# ❌ Error Loading Overview\n\n{str(e)}"
                
                try:
                    metrics = self.get_detailed_metrics()
                except Exception as e:
                    metrics = f"## ❌ Error Loading Metrics\n\n{str(e)}"
                
                try:
                    capabilities = self.get_api_capabilities()
                except Exception as e:
                    capabilities = f"## ❌ Error Loading Capabilities\n\n{str(e)}"
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return overview, metrics, capabilities, timestamp
            
            def refresh_api_capabilities_only():
                """Refresh only API capabilities"""
                try:
                    print("🔄 Refreshing API capabilities...")
                    capabilities = self.get_api_capabilities()
                    print("✅ API capabilities refreshed successfully")
                    return capabilities
                except Exception as e:
                    error_msg = f"## ❌ Error Loading API Capabilities\n\n{str(e)}"
                    print(f"❌ API capabilities refresh failed: {str(e)}")
                    return error_msg
            
            def refresh_diagnostics_only():
                """Refresh only diagnostics information"""
                try:
                    print("🔄 Refreshing diagnostics...")
                    diagnostics = self.run_diagnostics()
                    print("✅ Diagnostics refreshed successfully")
                    return diagnostics
                except Exception as e:
                    error_msg = f"Error running diagnostics: {str(e)}"
                    print(f"❌ Diagnostics refresh failed: {str(e)}")
                    return error_msg
            
            # Metrics collection handlers
            def start_metrics_collection(interval):
                """Start metrics collection with specified interval"""
                try:
                    self.metrics_collector.set_pull_interval(int(interval))
                    self.metrics_collector.start_collection(self.client)
                    return "✅ Collection started"
                except Exception as e:
                    return f"❌ Error starting collection: {str(e)}"
            
            def stop_metrics_collection():
                """Stop metrics collection"""
                try:
                    self.metrics_collector.stop_collection()
                    return "⏸️ Collection stopped"
                except Exception as e:
                    return f"❌ Error stopping collection: {str(e)}"
            
            def update_pull_interval(interval):
                """Update the pull interval for metrics collection"""
                try:
                    self.metrics_collector.set_pull_interval(int(interval))
                    return f"⏱️ Interval updated to {interval}s"
                except Exception as e:
                    return f"❌ Error updating interval: {str(e)}"
            
            def refresh_metrics_plots():
                """Refresh all enhanced metrics dashboards with archive loading if needed"""
                try:
                    print("🔄 Refreshing enhanced metrics dashboards...")
                    
                    # Check if we have any data, if not try to load from archive
                    total_metrics = len(self.metrics_collector.metrics_data)
                    total_timestamps = len(self.metrics_collector.timestamps)
                    
                    if total_metrics == 0 and total_timestamps == 0:
                        print("📦 No data found, attempting to load from archive...")
                        try:
                            self.metrics_collector.load_archive()
                            total_metrics = len(self.metrics_collector.metrics_data)
                            total_timestamps = len(self.metrics_collector.timestamps)
                            if total_metrics > 0:
                                print(f"✅ Loaded {total_metrics} metrics from archive")
                        except Exception as archive_e:
                            print(f"⚠️ Archive load failed: {str(archive_e)}")
                    
                    dashboards = self.get_metrics_plots()
                    
                    print(f"📊 Retrieved dashboards: {list(dashboards.keys())}")
                    
                    # Extract dashboards for the 4 new actionable categories
                    performance_dash = dashboards.get('performance')
                    health_dash = dashboards.get('health') 
                    efficiency_dash = dashboards.get('efficiency')
                    insights_dash = dashboards.get('insights')
                    
                    print(f"⚡ Performance dashboard: {'✅ Present' if performance_dash else '❌ None'}")
                    print(f"🏥 Health dashboard: {'✅ Present' if health_dash else '❌ None'}")
                    print(f"🚀 Efficiency dashboard: {'✅ Present' if efficiency_dash else '❌ None'}")
                    print(f"🧠 Insights dashboard: {'✅ Present' if insights_dash else '❌ None'}")
                    
                    metrics_summary = self.get_detailed_metrics()
                    
                    return (
                        performance_dash,
                        health_dash,
                        efficiency_dash,
                        insights_dash,
                        metrics_summary
                    )
                except Exception as e:
                    print(f"💥 Error in refresh_metrics_dashboards: {str(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    error_html = f"<div style='text-align: center; color: red; padding: 40px; background: #fff5f5; border-radius: 8px; margin: 20px;'>Error refreshing dashboards: {str(e)}</div>"
                    return (error_html, error_html, error_html, error_html, f"Error refreshing dashboards: {str(e)}")
            
            
            
            def debug_info():
                """Get debug information about the plotting system"""
                try:
                    debug_msg = "🔧 Debug Information\n\n"
                    
                    # Check Plotly availability
                    debug_msg += f"**Plotly Available:** {PLOTTING_AVAILABLE}\n"
                    
                    if PLOTTING_AVAILABLE:
                        try:
                            import plotly
                            debug_msg += f"**Plotly Version:** {plotly.__version__}\n"
                        except:
                            debug_msg += "**Plotly Version:** Could not determine\n"
                    
                    # Check metrics data
                    total_metrics = len(self.metrics_collector.metrics_data)
                    debug_msg += f"**Total Metrics Collected:** {total_metrics}\n"
                    debug_msg += f"**Timestamps Available:** {len(self.metrics_collector.timestamps)}\n"
                    
                    # Show actual metric names that were collected
                    debug_msg += "\n**📊 Collected Metric Names:**\n"
                    metric_names = list(self.metrics_collector.metrics_data.keys())
                    for i, name in enumerate(metric_names[:20]):  # Show first 20
                        category = self.metrics_collector.categorize_metric(name)
                        debug_msg += f"- `{name}` → {category}\n"
                    if len(metric_names) > 20:
                        debug_msg += f"- ... and {len(metric_names) - 20} more\n"
                    
                    # Check categories
                    debug_msg += "\n**📈 Category Breakdown:**\n"
                    for category in ['memory', 'transactions', 'tokens', 'model']:
                        cat_data = self.metrics_collector.get_metrics_by_category(category)
                        debug_msg += f"**{category.title()}:** {len(cat_data)} metrics\n"
                        for metric_name in list(cat_data.keys())[:3]:  # Show first 3 per category
                            debug_msg += f"  - {metric_name}\n"
                    
                    # Count uncategorized metrics
                    uncategorized_count = 0
                    uncategorized_names = []
                    for name in metric_names:
                        if self.metrics_collector.categorize_metric(name) == 'other':
                            uncategorized_count += 1
                            if len(uncategorized_names) < 5:
                                uncategorized_names.append(name)
                    
                    debug_msg += f"\n**❓ Uncategorized:** {uncategorized_count} metrics\n"
                    for name in uncategorized_names:
                        debug_msg += f"  - {name}\n"
                    
                    # Try creating a simple plot
                    if PLOTTING_AVAILABLE:
                        try:
                            import plotly.graph_objects as go
                            test_fig = go.Figure()
                            test_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], name='Test'))
                            debug_msg += "\n**Test Plot Creation:** ✅ Success\n"
                        except Exception as plot_error:
                            debug_msg += f"\n**Test Plot Creation:** ❌ Failed - {str(plot_error)}\n"
                    
                    # Add detailed plot debugging
                    debug_msg += "\n\n**🎯 PLOT DEBUGGING:**\n"
                    try:
                        for category in ['memory', 'transactions', 'tokens', 'model']:
                            debug_msg += f"\n**{category.title()} Plot Debug:**\n"
                            cat_data = self.metrics_collector.get_metrics_by_category(category)
                            debug_msg += f"  - Found {len(cat_data)} categorized metrics\n"
                            if cat_data:
                                for metric_name, data in list(cat_data.items())[:2]:  # Show first 2
                                    values = data.get('values', [])
                                    timestamps = data.get('timestamps', [])
                                    debug_msg += f"  - {metric_name}: {len(values)} values, {len(timestamps)} timestamps\n"
                                
                                # Try creating the plot
                                try:
                                    fig = self.metrics_collector.create_time_series_plot(category)
                                    if fig:
                                        debug_msg += f"  - Plot creation: ✅ SUCCESS\n"
                                    else:
                                        debug_msg += f"  - Plot creation: ❌ RETURNED NONE\n"
                                except Exception as plot_err:
                                    debug_msg += f"  - Plot creation: ❌ ERROR: {str(plot_err)}\n"
                            else:
                                debug_msg += f"  - No categorized data found\n"
                    except Exception as plot_debug_err:
                        debug_msg += f"❌ Plot debug error: {str(plot_debug_err)}\n"
                    
                    return debug_msg
                    
                except Exception as e:
                    return f"❌ Debug error: {str(e)}"
            
            # Archive management handlers
            def export_data(filename):
                """Export metrics data"""
                try:
                    result = self.metrics_collector.export_metrics(filename if filename.strip() else None)
                    return result
                except Exception as e:
                    return f"❌ Export error: {str(e)}"
            
            def import_data(selected_file):
                """Import metrics data"""
                try:
                    if not selected_file:
                        return "❌ Please select a file to import"
                    result = self.metrics_collector.import_metrics(selected_file)
                    # Refresh plots after import
                    plots = self.get_metrics_plots()
                    return result
                except Exception as e:
                    return f"❌ Import error: {str(e)}"
            
            def clear_archive():
                """Clear archived data"""
                try:
                    with self.metrics_collector.lock:
                        self.metrics_collector.metrics_data.clear()
                        self.metrics_collector.timestamps.clear()
                    
                    # Remove archive file
                    if self.metrics_collector.archive_file.exists():
                        self.metrics_collector.archive_file.unlink()
                    
                    return "✅ Archive cleared successfully"
                except Exception as e:
                    return f"❌ Clear error: {str(e)}"
            
            def refresh_import_files():
                """Refresh list of available import files"""
                try:
                    archive_dir = self.metrics_collector.archive_dir
                    json_files = []
                    
                    if archive_dir.exists():
                        # Get all JSON files in archive directory
                        for file_path in archive_dir.glob("*.json"):
                            if file_path.name != "metrics_data.json":  # Skip current archive
                                json_files.append(file_path.name)
                    
                    return gr.update(choices=sorted(json_files))
                except Exception as e:
                    print(f"❌ Error refreshing files: {str(e)}")
                    return gr.update(choices=[])
            
            
            refresh_overview_btn.click(
                refresh_management_overview,
                outputs=[management_overview, metrics_output, capabilities_output, last_update_display]
            )
            
            # Individual tab refresh handlers
            refresh_capabilities_btn.click(
                refresh_api_capabilities_only,
                outputs=[capabilities_output]
            )
            
            refresh_diagnostics_btn.click(
                refresh_diagnostics_only,
                outputs=[diagnostics_output]
            )
            
            run_diagnostics_btn.click(
                refresh_diagnostics_only,
                outputs=[diagnostics_output]
            )
            
            # Wire up metrics collection controls
            start_collection_btn.click(
                start_metrics_collection,
                inputs=[pull_interval_slider],
                outputs=[collection_status]
            )
            
            stop_collection_btn.click(
                stop_metrics_collection,
                outputs=[collection_status]
            )
            
            pull_interval_slider.change(
                update_pull_interval,
                inputs=[pull_interval_slider],
                outputs=[collection_status]
            )
            
            
            # Debug information
            debug_btn.click(
                debug_info,
                outputs=[collection_status]
            )
            
            
            # Manual refresh plots for enhanced dashboard
            refresh_plots_btn.click(
                refresh_metrics_plots,
                outputs=[performance_dashboard, health_dashboard, efficiency_dashboard, insights_dashboard, metrics_output]
            )
            
            
            # Archive management handlers
            export_btn.click(
                export_data,
                inputs=[export_filename],
                outputs=[archive_status]
            )
            
            import_btn.click(
                import_data,
                inputs=[import_file_dropdown],
                outputs=[archive_status]
            )
            
            clear_archive_btn.click(
                clear_archive,
                outputs=[archive_status]
            )
            
            refresh_files_btn.click(
                refresh_import_files,
                outputs=[import_file_dropdown]
            )
            
            # Auto-refresh plots every 15 seconds when collection is active
            plots_timer = gr.Timer(15, active=False)
            plots_timer.tick(
                refresh_metrics_plots,
                outputs=[performance_dashboard, health_dashboard, efficiency_dashboard, insights_dashboard, metrics_output]
            )
            
            # Auto-load data on interface load
            def auto_load_interface_data():
                """Auto-load data when interface starts"""
                try:
                    print("🚀 Auto-loading interface data...")
                    
                    # Load archived metrics if available
                    try:
                        self.metrics_collector.load_archive()
                        archived_metrics = len(self.metrics_collector.metrics_data)
                        archived_timestamps = len(self.metrics_collector.timestamps)
                        if archived_metrics > 0:
                            print(f"📦 Loaded {archived_metrics} archived metrics")
                    except Exception as e:
                        print(f"⚠️ Archive load failed: {str(e)}")
                        archived_metrics = 0
                    
                    # Get initial overview data
                    try:
                        overview = self.get_management_overview()
                    except Exception as e:
                        overview = f"# ❌ Error Loading Overview\n\n{str(e)}"
                    
                    # Get initial capabilities data
                    try:
                        capabilities = self.get_api_capabilities()
                    except Exception as e:
                        capabilities = f"## ❌ Error Loading Capabilities\n\n{str(e)}"
                    
                    # Get initial metrics summary
                    try:
                        metrics = self.get_detailed_metrics()
                    except Exception as e:
                        metrics = f"## ❌ Error Loading Metrics\n\n{str(e)}"
                    
                    # Get initial enhanced dashboards if we have data (consolidated structure)
                    scheduler_dash = performance_dash = cache_dash = advanced_dash = None
                    
                    if archived_metrics > 0:
                        try:
                            print("🎨 Generating initial consolidated dashboards...")
                            dashboards = self.get_metrics_plots()
                            scheduler_dash = dashboards.get('scheduler', '<div>No scheduler data</div>')
                            performance_dash = dashboards.get('performance', '<div>No performance data</div>')
                            cache_dash = dashboards.get('cache', '<div>No cache data</div>')
                            advanced_dash = dashboards.get('advanced', '<div>No advanced data</div>')
                            print("✅ Initial consolidated dashboards loaded successfully")
                        except Exception as e:
                            print(f"⚠️ Initial dashboard loading failed: {str(e)}")
                            # Set default placeholders
                            default_placeholder = "<div style='text-align: center; padding: 40px; color: #666;'>Loading metrics data...</div>"
                            scheduler_dash = performance_dash = cache_dash = advanced_dash = default_placeholder
                    else:
                        # No archived data - set loading placeholders  
                        loading_placeholder = "<div style='text-align: center; padding: 40px; color: #666;'>No metrics data available. Start collection or refresh to load data.</div>"
                        scheduler_dash = performance_dash = cache_dash = advanced_dash = loading_placeholder
                    
                    # Initial collection status
                    archive_msg = ""
                    if archived_metrics > 0:
                        archive_msg = f" (📦 Loaded {archived_metrics} archived metrics)"
                        collection_msg = f"✅ Interface loaded with {archived_metrics} metrics{archive_msg}"
                    else:
                        collection_msg = "🔄 Interface loaded - start collection to see metrics"
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    return (
                        overview,
                        metrics, 
                        capabilities,
                        timestamp,
                        collection_msg,
                        scheduler_dash,
                        performance_dash,
                        cache_dash,
                        advanced_dash
                    )
                    
                except Exception as e:
                    print(f"❌ Auto-load failed: {str(e)}")
                    error_msg = f"❌ Interface load error: {str(e)}"
                    error_dash = "<div style='text-align: center; color: red; padding: 40px;'>Error loading dashboard</div>"
                    return (
                        error_msg, error_msg, error_msg, 
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        error_msg, 
                        error_dash, error_dash, error_dash, error_dash
                    )
            
            # Legacy function for backward compatibility  
            def auto_start_collection():
                """Auto-start metrics collection when interface loads (legacy)"""
                try:
                    # Check if archive was loaded
                    archived_metrics = len(self.metrics_collector.metrics_data)
                    archived_timestamps = len(self.metrics_collector.timestamps)
                    
                    archive_msg = ""
                    if archived_metrics > 0:
                        archive_msg = f" (📦 Loaded {archived_metrics} archived metrics)"
                    
                    self.metrics_collector.set_pull_interval(30)  # 30 second default
                    self.metrics_collector.start_collection(self.client)
                    
                    # Generate initial tables
                    dashboards = self.get_metrics_plots()
                    
                    status_message = f"⚡ Auto-started metrics collection{archive_msg}"
                    
                    return (
                        overview, 
                        self.get_detailed_metrics(),
                        capabilities,
                        timestamp,
                        status_message,
                        dashboards.get('performance', '<div>No performance dashboard</div>'),
                        dashboards.get('health', '<div>No health dashboard</div>'), 
                        dashboards.get('efficiency', '<div>No efficiency dashboard</div>'),
                        dashboards.get('insights', '<div>No insights dashboard</div>')
                    )
                except Exception as e:
                    error_dash = f"<div style='text-align: center; color: red; padding: 40px;'>Error loading dashboard: {str(e)}</div>"
                    return None, f"❌ Auto-start failed: {str(e)}", str(e), "Archive load failed", "Not started", error_dash, error_dash, error_dash, error_dash
            
            # Initialize interface data on startup
            interface.load(
                auto_load_interface_data,
                outputs=[management_overview, metrics_output, capabilities_output, last_update_display, 
                         collection_status, performance_dashboard, health_dashboard, efficiency_dashboard, insights_dashboard]
            )
            
            # Auto-refresh functionality with timer
            timer = gr.Timer(30, active=False)  # 30 second timer, inactive by default
            
            # Auto-refresh handler
            timer.tick(
                refresh_management_overview,
                outputs=[management_overview, metrics_output, capabilities_output, last_update_display]
            )
            
            # Control timer based on checkbox
            def update_timer(active):
                """Update timer active state"""
                if hasattr(timer, 'active'):
                    timer.active = active
                return timer
            
            auto_refresh_checkbox.change(
                update_timer,
                inputs=[auto_refresh_checkbox],
                outputs=[timer]
            )
            
            # Legacy diagnostics handler (Quick Actions panel)
            diagnostics_btn.click(
                refresh_diagnostics_only,
                outputs=[diagnostics_output]
            )
            
            test_streaming_btn.click(
                self.test_simple_streaming,
                outputs=[diagnostics_output]
            )
            
            test_ui_btn.click(
                self.test_ui_update,
                inputs=[chatbot],
                outputs=[msg, chatbot, file_upload]
            )
            
            # Message handling with session management
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot, system_dropdown, custom_system, 
                       temperature, max_tokens, file_upload, session_id_input],
                outputs=[msg, chatbot, file_upload, session_id_input],
                show_progress="minimal"
            )
            
            submit.click(
                self.process_message,
                inputs=[msg, chatbot, system_dropdown, custom_system,
                       temperature, max_tokens, file_upload, session_id_input],
                outputs=[msg, chatbot, file_upload, session_id_input],
                show_progress="minimal"
            )
            
            # Clear history
            clear.click(
                self.clear_history,
                outputs=[chatbot]
            )
            
            # Export conversation
            def handle_export(history):
                export_text = self.export_conversation(history)
                return export_text, gr.update(visible=True)
            
            export.click(
                handle_export,
                inputs=[chatbot],
                outputs=[export_output, export_output]
            )
            
            # Prompt management handlers
            def reload_prompts_handler():
                result = self.reload_system_prompts()
                # Update all dropdown choices
                choices = list(self.system_prompts.keys())
                return (
                    result, 
                    gr.update(choices=choices),
                    gr.update(choices=choices, value=choices[0] if choices else None)
                )
            
            reload_prompts_btn.click(
                reload_prompts_handler,
                outputs=[prompt_status, system_dropdown, existing_prompt_dropdown]
            )
            
            # Load prompt for editing
            def load_prompt_handler(prompt_name):
                name, content = self.load_prompt_for_editing(prompt_name)
                status = f"Loaded prompt '{prompt_name}' for editing" if name else "No prompt selected"
                return name, content, status
            
            load_prompt_btn.click(
                load_prompt_handler,
                inputs=[existing_prompt_dropdown],
                outputs=[edit_prompt_name, edit_prompt_content, prompt_status]
            )
            
            # Save/update prompt
            def save_prompt_handler(name, content):
                result = self.save_custom_prompt(name, content)
                # Update dropdown choices if successful
                if "✅" in result:
                    choices = list(self.system_prompts.keys())
                    return (
                        result, 
                        gr.update(choices=choices),
                        gr.update(choices=choices, value=name),
                        name,  # Keep name
                        content  # Keep content
                    )
                return result, gr.update(), gr.update(), name, content
            
            save_prompt_btn.click(
                save_prompt_handler,
                inputs=[edit_prompt_name, edit_prompt_content],
                outputs=[prompt_status, system_dropdown, existing_prompt_dropdown, edit_prompt_name, edit_prompt_content]
            )
            
            # Delete prompt
            def delete_prompt_handler(prompt_name):
                result = self.delete_prompt(prompt_name)
                if "✅" in result:
                    choices = list(self.system_prompts.keys())
                    return (
                        result,
                        gr.update(choices=choices),
                        gr.update(choices=choices, value=choices[0] if choices else None),
                        "",  # Clear edit form
                        ""   # Clear edit form
                    )
                return result, gr.update(), gr.update(), gr.update(), gr.update()
            
            delete_prompt_btn.click(
                delete_prompt_handler,
                inputs=[existing_prompt_dropdown],
                outputs=[prompt_status, system_dropdown, existing_prompt_dropdown, edit_prompt_name, edit_prompt_content]
            )
            
            # Clear form
            def clear_form_handler():
                return "", "", "Form cleared - ready for new prompt"
            
            clear_form_btn.click(
                clear_form_handler,
                outputs=[edit_prompt_name, edit_prompt_content, prompt_status]
            )
            
            # Session management handlers
            def load_session_handler(session_id):
                if not session_id.strip():
                    return [], "Default Assistant", "", self.config.default_temperature, self.config.default_max_tokens, "Please enter a session ID"
                
                history, sys_prompt, custom_prompt, temp, tokens = self.load_session(session_id)
                status = f"Loaded session {session_id} with {len(history)} messages"
                return history, sys_prompt, custom_prompt, temp, tokens, status
            
            def new_session_handler():
                history, session_id = self.new_session()
                return history, session_id, f"Created new session: {session_id}"
            
            def list_sessions_handler():
                sessions_text, session_ids = self.get_session_list()
                # Create dropdown choices with session info
                dropdown_choices = []
                sessions = self.session_manager.list_sessions()
                for session in sessions[:10]:
                    created = datetime.fromtimestamp(session['created']).strftime('%m-%d %H:%M')
                    label = f"{session['id']} ({session['messages']} msgs, {created})"
                    dropdown_choices.append((label, session['id']))
                
                # Return for displays, dropdown update, and open accordion
                return (
                    sessions_text, 
                    sessions_text, 
                    gr.update(open=True),
                    gr.update(choices=dropdown_choices)
                )
            
            def cleanup_sessions_handler():
                cleaned = self.session_manager.cleanup_expired_sessions()
                return f"Cleaned up {cleaned} expired sessions"
            
            def quick_load_handler(selected_session_id):
                if not selected_session_id:
                    return [], "Default Assistant", "", self.config.default_temperature, self.config.default_max_tokens, selected_session_id, "Please select a session from the dropdown"
                
                history, sys_prompt, custom_prompt, temp, tokens = self.load_session(selected_session_id)
                status = f"✅ Quick loaded session {selected_session_id} ({len(history)} messages)"
                return history, sys_prompt, custom_prompt, temp, tokens, selected_session_id, status
            
            # Wire up session management
            load_session_btn.click(
                load_session_handler,
                inputs=[session_id_input],
                outputs=[chatbot, system_dropdown, custom_system, temperature, max_tokens, context_info]
            )
            
            new_session_btn.click(
                new_session_handler,
                outputs=[chatbot, session_id_input, context_info]
            )
            
            list_sessions_btn.click(
                list_sessions_handler,
                outputs=[sessions_display, sessions_info, sessions_accordion, session_dropdown]
            )
            
            
            cleanup_sessions_btn.click(
                cleanup_sessions_handler,
                outputs=[sessions_info]
            )
            
            # Quick load handler
            quick_load_btn.click(
                quick_load_handler,
                inputs=[session_dropdown],
                outputs=[chatbot, system_dropdown, custom_system, temperature, max_tokens, session_id_input, context_info]
            )
        
        return interface

def main():
    """Launch the enhanced chat interface"""
    print("🚀 Starting Telco-AIX SME Web Interface...")
    
    config = Config()
    chat = ChatInterface(config)
    interface = chat.create_interface()
    
    print(f"🌐 Launching on port {30180}...")
    interface.launch(
        auth=(config.admin_username, config.admin_password),
        server_name="0.0.0.0",
        server_port=30180,
        share=False,
        inbrowser=False,
        favicon_path=None,
        ssl_verify=False,
        show_api=False,  # Hide API documentation
        max_threads=40,  # Better concurrency
        quiet=False
    )

if __name__ == "__main__":
    main()
