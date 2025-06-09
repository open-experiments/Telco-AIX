// Global state
let currentWorkflowPage = 1;
let workflowsData = { workflows: [], pagination: {}, stats: {} };
let progressData = {};
let timelineInitialized = false;
let topologyData = {};
let selectedAgent = null;
let networkChart = null;
let dataFlowParticles = [];

// Real-time metric charts
let amfChart = null;
let smfChart = null;
let upfChart = null;
let metricsData = {
    amf: { anomalies: [] },
    smf: { anomalies: [] },
    upf: { anomalies: [] }
};

// Initialize metric charts
function initMetricCharts() {
    // AMF Chart
    const amfCtx = document.getElementById('amfMetricsChart');
    if (amfCtx) {
        amfChart = new Chart(amfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Registration Rate',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                }, {
                    label: 'Registration Success Rate',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }, {
                    label: 'Auth Success Rate',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }]
            },
            options: getChartOptions('Rate', 'Success Rate (%)')
        });
    }

    // SMF Chart
    const smfCtx = document.getElementById('smfMetricsChart');
    if (smfCtx) {
        smfChart = new Chart(smfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Session Est. Rate',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                }, {
                    label: 'Session Success Rate',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }, {
                    label: 'IP Pool Usage',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }]
            },
            options: getChartOptions('Rate', 'Percentage (%)')
        });
    }

    // UPF Chart
    const upfCtx = document.getElementById('upfMetricsChart');
    if (upfCtx) {
        upfChart = new Chart(upfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Active Sessions',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                }, {
                    label: 'Throughput (Mbps)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }, {
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    yAxisID: 'y2',
                    tension: 0.4
                }]
            },
            options: getChartOptions('Sessions', 'Throughput/Latency', true)
        });
    }
}

// Get chart options configuration
function getChartOptions(leftAxisTitle, rightAxisTitle, hasThirdAxis = false) {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false
        },
        plugins: {
            legend: {
                labels: { color: '#ffffff' }
            },
            tooltip: {
                callbacks: {
                    afterLabel: function(context) {
                        const dataIndex = context.dataIndex;
                        const component = context.chart.canvas.id.replace('MetricsChart', '');
                        const anomalies = metricsData[component].anomalies;
                        
                        // Check if this timestamp has an anomaly
                        const timestamp = metricsData[component].timestamps[dataIndex];
                        const hasAnomaly = anomalies.some(a => {
                            const anomalyTime = new Date(a.timestamp).getTime();
                            const dataTime = new Date(timestamp).getTime();
                            return Math.abs(anomalyTime - dataTime) < 60000; // Within 1 minute
                        });
                        
                        return hasAnomaly ? '⚠️ Anomaly detected' : '';
                    }
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#ffffff' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            },
            y: {
                type: 'linear',
                display: true,
                position: 'left',
                title: { display: true, text: leftAxisTitle, color: '#ffffff' },
                ticks: { color: '#ffffff' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            },
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: { display: true, text: rightAxisTitle, color: '#ffffff' },
                ticks: { color: '#ffffff' },
                grid: { drawOnChartArea: false }
            }
        }
    };

    if (hasThirdAxis) {
        options.scales.y2 = {
            type: 'linear',
            display: true,
            position: 'right',
            offset: true,
            title: { display: true, text: 'Latency (ms)', color: '#ffffff' },
            ticks: { color: '#ffffff' },
            grid: { drawOnChartArea: false }
        };
    }

    return options;
}

// Update metric charts with new data (deprecated - use fetchMetrics instead)
function updateMetricCharts() {
    // This function is deprecated
    // Metrics are now updated via fetchMetrics() and updateMetricsFromAPI()
}

// Add anomaly markers to charts
function addAnomalyToChart(component, timestamp, severity) {
    const componentLower = component.toLowerCase();
    if (metricsData[componentLower]) {
        metricsData[componentLower].anomalies.push({
            timestamp: timestamp,
            severity: severity
        });
    }
}

// Enhanced time bar management
function updateEnhancedTimeBar() {
    if (progressData.timeline_data) {
        const timelineData = progressData.timeline_data;
        const progressBar = document.getElementById('time-progress');
        const timelineProgress = document.getElementById('timeline-progress');
        const anomalyProgress = document.getElementById('anomaly-progress');
        const timeElapsed = document.getElementById('time-elapsed');
        const etaTime = document.getElementById('eta-time');
        
        // Update timeline progress (based on actual data timeline)
        const percentage = timelineData.timeline_progress_percentage || 0;
        if (progressBar) progressBar.style.width = percentage + '%';
        if (timelineProgress) timelineProgress.textContent = percentage.toFixed(1) + '%';
        
        // FIXED: Update anomaly progress with discovered counts
        if (anomalyProgress) {
            const timelinePos = progressData.current_timeline_position ? 
                new Date(progressData.current_timeline_position).toLocaleTimeString() : 'Starting...';
            anomalyProgress.textContent = `${progressData.discovered_anomalies || 0} / ${progressData.total_anomalies || 0} (${timelinePos})`;
        }
        
        // Update processing time
        if (timelineData.processing_elapsed_seconds && timeElapsed) {
            const elapsed = timelineData.processing_elapsed_seconds;
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = Math.floor(elapsed % 60);
            timeElapsed.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        // Update ETA
        if (timelineData.eta_seconds !== undefined && etaTime) {
            if (timelineData.eta_seconds > 0) {
                const eta = timelineData.eta_seconds;
                const etaHours = Math.floor(eta / 3600);
                const etaMinutes = Math.floor((eta % 3600) / 60);
                const etaSecondsRem = Math.floor(eta % 60);
                etaTime.textContent = `${etaHours.toString().padStart(2, '0')}:${etaMinutes.toString().padStart(2, '0')}:${etaSecondsRem.toString().padStart(2, '0')}`;
            } else {
                etaTime.textContent = 'Complete';
            }
        }
        
        // Update speed status
        const speedStatus = document.getElementById('speed-status');
        if (speedStatus && progressData.processing_speed) {
            const speed = progressData.processing_speed;
            const speedText = speed === 1 ? 'Normal Speed' : 
                            speed === 5 ? 'Fast (5x)' :
                            speed === 10 ? 'Very Fast (10x)' :
                            speed === 100 ? 'Ultra Fast (100x)' : `${speed}x Speed`;
            speedStatus.textContent = speedText;
            
            // Update active speed button
            document.querySelectorAll('.speed-button').forEach(btn => {
                btn.classList.remove('active');
                if (parseInt(btn.dataset.speed) === speed) {
                    btn.classList.add('active');
                }
            });
        }
    }
}

// Speed control functions
async function setProcessingSpeed(speed) {
    try {
        const response = await fetch(`/api/speed/${speed}`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        
        console.log(`Speed set to ${speed}x:`, result);
        
        // Update UI immediately
        const speedStatus = document.getElementById('speed-status');
        if (speedStatus) {
            const speedText = speed === 1 ? 'Normal Speed' : 
                            speed === 5 ? 'Fast (5x)' :
                            speed === 10 ? 'Very Fast (10x)' :
                            speed === 100 ? 'Ultra Fast (100x)' : `${speed}x Speed`;
            speedStatus.textContent = speedText;
        }
        
        // Update active button
        document.querySelectorAll('.speed-button').forEach(btn => {
            btn.classList.remove('active');
            if (parseInt(btn.dataset.speed) === speed) {
                btn.classList.add('active');
            }
        });
        
        // Force refresh progress data
        setTimeout(fetchProgress, 500);
        
    } catch (error) {
        console.error('Error setting processing speed:', error);
    }
}

// Initialize enhanced agent topology
function initEnhancedAgentTopology() {
    const topology = document.getElementById('enhanced-agent-topology');
    if (!topology) return;
    
    const agents = [
        {id: 'diagnostic', x: 80, y: 80, color: '#3b82f6'},
        {id: 'planning', x: 320, y: 80, color: '#10b981'},
        {id: 'execution', x: 560, y: 80, color: '#f59e0b'},
        {id: 'validation', x: 320, y: 280, color: '#ef4444'}
    ];
    
    // Clear existing content
    topology.innerHTML = '';
    
    // Create enhanced agent nodes
    agents.forEach(agent => {
        const nodeEl = createEnhancedAgentNode(agent);
        topology.appendChild(nodeEl);
    });
    
    // Create connections
    createEnhancedConnections(topology);
    
    // Initialize network analytics chart
    initNetworkChart();
    
    // Start data flow animation
    startDataFlowAnimation();
}

function createEnhancedAgentNode(agent) {
    const nodeEl = document.createElement('div');
    nodeEl.className = 'enhanced-agent-node';
    nodeEl.id = `enhanced-agent-${agent.id}`;
    nodeEl.style.left = agent.x + 'px';
    nodeEl.style.top = agent.y + 'px';
    nodeEl.style.background = `linear-gradient(135deg, ${agent.color}, ${adjustBrightness(agent.color, -20)})`;
    
    nodeEl.innerHTML = `
        <div style="font-size: 0.9em; font-weight: 700;">${agent.id.toUpperCase()}</div>
        <div class="agent-metrics-overlay" id="metrics-${agent.id}">
            <div>CPU: 0%</div>
            <div>Queue: 0</div>
        </div>
    `;
    
    // Add click handler for detailed view
    nodeEl.addEventListener('click', () => showAgentDetails(agent.id));
    
    return nodeEl;
}

// Fixed createEnhancedConnections function with proper alignment
function createEnhancedConnections(topology) {
    // Agent node dimensions
    const nodeWidth = 100;
    const nodeHeight = 100;
    const nodeRadius = nodeWidth / 2;
    
    // Agent positions (from the original agents array)
    const agentPositions = {
        'diagnostic': { x: 80, y: 80 },
        'planning': { x: 320, y: 80 },
        'execution': { x: 560, y: 80 },
        'validation': { x: 320, y: 280 }
    };
    
    // Calculate center points for each agent
    const centers = {};
    Object.keys(agentPositions).forEach(agentId => {
        centers[agentId] = {
            x: agentPositions[agentId].x + nodeRadius,
            y: agentPositions[agentId].y + nodeRadius
        };
    });
    
    // Define connections using proper center coordinates
    const connections = [
        {
            from: centers.diagnostic,
            to: centers.planning,
            id: 'diagnostic-planning'
        },
        {
            from: centers.planning,
            to: centers.execution,
            id: 'planning-execution'
        },
        {
            from: centers.execution,
            to: centers.validation,
            id: 'execution-validation'
        },
        {
            from: centers.validation,
            to: centers.diagnostic,
            id: 'validation-diagnostic'
        }
    ];
    
    connections.forEach(conn => {
        const connEl = createEnhancedConnection(conn);
        topology.appendChild(connEl);
    });
}

// Enhanced connection creation with better positioning
function createEnhancedConnection(conn) {
    const length = Math.sqrt(Math.pow(conn.to.x - conn.from.x, 2) + Math.pow(conn.to.y - conn.from.y, 2));
    const angle = Math.atan2(conn.to.y - conn.from.y, conn.to.x - conn.from.x) * 180 / Math.PI;
    
    const connEl = document.createElement('div');
    connEl.className = 'agent-connection enhanced';
    connEl.id = `connection-${conn.id}`;
    connEl.style.left = conn.from.x + 'px';
    connEl.style.top = conn.from.y + 'px';
    connEl.style.width = length + 'px';
    connEl.style.height = '3px';
    connEl.style.background = 'linear-gradient(90deg, rgba(59, 130, 246, 0.8), rgba(16, 185, 129, 0.8))';
    connEl.style.transform = `rotate(${angle}deg)`;
    connEl.style.transformOrigin = '0 50%';
    connEl.style.position = 'absolute';
    connEl.style.borderRadius = '2px';
    connEl.style.opacity = '0.7';
    connEl.style.boxShadow = '0 0 8px rgba(59, 130, 246, 0.3)';
    connEl.style.zIndex = '1';
    
    return connEl;
}

function showAgentDetails(agentId) {
    selectedAgent = agentId;
    const panel = document.getElementById('agent-details-panel');
    const content = document.getElementById('panel-content');
    const nameEl = document.getElementById('panel-agent-name');
    
    if (topologyData.agents && topologyData.agents[agentId]) {
        const agent = topologyData.agents[agentId];
        
        nameEl.textContent = `${agentId.toUpperCase()} Agent Details`;
        
        content.innerHTML = `
            <div class="performance-metric">
                <span>Health Score:</span>
                <span class="performance-value">${agent.health_score.toFixed(1)}%</span>
            </div>
            <div class="performance-metric">
                <span>CPU Usage:</span>
                <div class="performance-bar">
                    <div class="performance-fill" style="width: ${agent.cpu_usage}%"></div>
                </div>
                <span class="performance-value">${agent.cpu_usage.toFixed(1)}%</span>
            </div>
            <div class="performance-metric">
                <span>Memory Usage:</span>
                <div class="performance-bar">
                    <div class="performance-fill" style="width: ${agent.memory_usage}%"></div>
                </div>
                <span class="performance-value">${agent.memory_usage.toFixed(1)}%</span>
            </div>
            <div class="performance-metric">
                <span>Task Queue:</span>
                <span class="performance-value">${agent.task_queue_size}</span>
            </div>
            <div class="performance-metric">
                <span>Throughput/min:</span>
                <span class="performance-value">${agent.tasks_processed_per_minute.toFixed(1)}</span>
            </div>
            <div class="performance-metric">
                <span>Avg Response Time:</span>
                <span class="performance-value">${agent.average_response_time.toFixed(0)}ms</span>
            </div>
            <div class="performance-metric">
                <span>Error Rate:</span>
                <span class="performance-value">${agent.error_rate.toFixed(2)}%</span>
            </div>
            
            <h5 style="margin: 20px 0 10px 0; color: #3b82f6;">Recent Performance Trend</h5>
            <div style="display: flex; align-items: end; height: 60px; gap: 2px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                ${agent.cpu_history && agent.cpu_history.length > 0 ? 
                    agent.cpu_history.map(cpu => `<div style="width: 4px; height: ${cpu}%; background: #3b82f6; border-radius: 2px;"></div>`).join('') :
                    '<div style="color: #9ca3af;">No historical data available</div>'
                }
            </div>
        `;
        
        panel.style.display = 'block';
    }
}

function closeAgentPanel() {
    document.getElementById('agent-details-panel').style.display = 'none';
    selectedAgent = null;
}

function updateEnhancedAgentTopology(data) {
    topologyData = data;
    
    // Update topology health metrics
    if (data.topology_health) {
        document.getElementById('topology-health').textContent = data.topology_health.overall_health.toFixed(1) + '%';
        document.getElementById('total-throughput').textContent = data.topology_health.total_throughput.toFixed(0);
    }
    
    if (data.network_stats) {
        document.getElementById('network-latency').textContent = data.network_stats.average_latency.toFixed(1) + 'ms';
        document.getElementById('message-success-rate').textContent = data.network_stats.success_rate.toFixed(1) + '%';
    }
    
    // Update agent nodes
    Object.keys(data.agents).forEach(agentId => {
        const agent = data.agents[agentId];
        const nodeEl = document.getElementById(`enhanced-agent-${agentId}`);
        const metricsEl = document.getElementById(`metrics-${agentId}`);
        
        if (nodeEl && metricsEl) {
            // Update metrics overlay
            metricsEl.innerHTML = `
                <div>CPU: ${agent.cpu_usage.toFixed(0)}%</div>
                <div>Queue: ${agent.task_queue_size}</div>
            `;
            
            // Update node status based on health
            nodeEl.classList.remove('critical', 'warning');
            if (agent.health_score < 60) {
                nodeEl.classList.add('critical');
            } else if (agent.health_score < 80) {
                nodeEl.classList.add('warning');
            }
        }
    });
    
    // Update performance grid
    updateAgentPerformanceGrid(data.agents);
    
    // Update communication log
    updateCommunicationLog(data.communications);
    
    // Update decision analytics
    updateDecisionAnalytics(data.decisions);
    
    // Update bottleneck predictions
    updateBottleneckPredictions(data.bottleneck_predictions);
    
    // Update network chart
    if (networkChart && data.network_stats) {
        updateNetworkChart(data.network_stats);
    }
}

function updateAgentPerformanceGrid(agents) {
    const grid = document.getElementById('agent-performance-grid');
    if (!grid) return;
    
    grid.innerHTML = Object.keys(agents).map(agentId => {
        const agent = agents[agentId];
        return `
            <div class="agent-performance-card">
                <h5 style="color: #3b82f6; margin-bottom: 10px;">${agentId.toUpperCase()}</h5>
                <div style="display: grid; gap: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Health:</span>
                        <span class="performance-value">${agent.health_score.toFixed(1)}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>CPU:</span>
                        <span class="performance-value">${agent.cpu_usage.toFixed(1)}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Memory:</span>
                        <span class="performance-value">${agent.memory_usage.toFixed(1)}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Queue:</span>
                        <span class="performance-value">${agent.task_queue_size}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function updateCommunicationLog(communications) {
    const log = document.getElementById('communication-log');
    if (!log || !communications) return;
    
    log.innerHTML = communications.slice(-10).map(comm => `
        <div class="communication-entry">
            <span>${comm.from_agent} → ${comm.to_agent}</span>
            <span style="color: ${comm.status === 'success' ? '#10b981' : '#ef4444'};">
                ${comm.latency_ms.toFixed(1)}ms
            </span>
        </div>
    `).join('');
}

function updateDecisionAnalytics(decisions) {
    const analytics = document.getElementById('decision-analytics');
    if (!analytics || !decisions) return;
    
    const avgConfidence = decisions.length > 0 
        ? decisions.reduce((sum, d) => sum + d.confidence_score, 0) / decisions.length * 100
        : 0;
    
    document.getElementById('decisions-count').textContent = decisions.length;
    document.getElementById('avg-confidence').textContent = avgConfidence.toFixed(1) + '%';
    
    const log = document.getElementById('decision-log');
    log.innerHTML = decisions.slice(-5).map(decision => `
        <div class="decision-entry">
            <div style="font-weight: 600;">${decision.agent_id.toUpperCase()}: ${decision.decision_type}</div>
            <div style="font-size: 0.8em; color: #9ca3af; margin-top: 3px;">
                Confidence: ${(decision.confidence_score * 100).toFixed(1)}%
            </div>
            <div style="font-size: 0.8em; margin-top: 3px;">${decision.reasoning}</div>
        </div>
    `).join('');
}

function updateBottleneckPredictions(predictions) {
    const container = document.getElementById('bottleneck-predictions');
    if (!container || !predictions) return;
    
    container.innerHTML = Object.keys(predictions).map(agentId => {
        const pred = predictions[agentId];
        const riskClass = pred.risk_score > 70 ? 'critical' : pred.risk_score > 40 ? 'warning' : 'normal';
        const riskColor = pred.risk_score > 70 ? 'risk-high' : pred.risk_score > 40 ? 'risk-medium' : 'risk-low';
        
        return `
            <div class="bottleneck-card ${riskClass}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h5>${agentId.toUpperCase()}</h5>
                    <span class="risk-score ${riskColor}">Risk: ${pred.risk_score.toFixed(0)}%</span>
                </div>
                ${pred.predicted_issues.length > 0 ? `
                    <div style="margin-bottom: 8px;">
                        <strong>Predicted Issues:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            ${pred.predicted_issues.map(issue => `<li>${issue}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                ${pred.recommendations.length > 0 ? `
                    <div>
                        <strong>Recommendations:</strong>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            ${pred.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function initNetworkChart() {
    const ctx = document.getElementById('networkChart');
    if (!ctx) return;
    
    networkChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => ''),
            datasets: [{
                label: 'Network Latency (ms)',
                data: Array.from({length: 20}, () => 0),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Throughput (KB/s)',
                data: Array.from({length: 20}, () => 0),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Latency (ms)', color: '#ffffff' },
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Throughput (KB/s)', color: '#ffffff' },
                    ticks: { color: '#ffffff' },
                    grid: { drawOnChartArea: false }
                },
                x: {
                    ticks: { color: '#ffffff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                }
            },
            animation: { duration: 500 }
        }
    });
}

function updateNetworkChart(networkStats) {
    if (!networkChart) return;
    
    // Add new data points
    networkChart.data.datasets[0].data.push(networkStats.average_latency);
    networkChart.data.datasets[1].data.push(networkStats.data_throughput / 1024); // Convert to KB
    
    // Remove old data points (keep last 20)
    if (networkChart.data.datasets[0].data.length > 20) {
        networkChart.data.datasets[0].data.shift();
        networkChart.data.datasets[1].data.shift();
    }
    
    networkChart.update('none'); // No animation for smooth real-time updates
}

function startDataFlowAnimation() {
    setInterval(() => {
        if (topologyData.communications && topologyData.communications.length > 0) {
            // Create data flow particles for recent successful communications
            const recentComms = topologyData.communications.filter(c => 
                c.status === 'success' && (Date.now() - new Date(c.timestamp).getTime()) < 30000
            );
            
            recentComms.forEach(comm => {
                if (Math.random() < 0.3) { // 30% chance to show particle
                    createDataFlowParticle(comm.from_agent, comm.to_agent);
                }
            });
        }
    }, 2000);
}

function createDataFlowParticle(fromAgent, toAgent) {
    const topology = document.getElementById('enhanced-agent-topology');
    if (!topology) return;
    
    const fromEl = document.getElementById(`enhanced-agent-${fromAgent}`);
    const toEl = document.getElementById(`enhanced-agent-${toAgent}`);
    if (!fromEl || !toEl) return;
    
    const particle = document.createElement('div');
    particle.className = 'data-flow-particle';
    
    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();
    const topologyRect = topology.getBoundingClientRect();
    
    const startX = fromRect.left - topologyRect.left + fromRect.width / 2;
    const startY = fromRect.top - topologyRect.top + fromRect.height / 2;
    const endX = toRect.left - topologyRect.left + toRect.width / 2;
    const endY = toRect.top - topologyRect.top + toRect.height / 2;
    
    particle.style.left = startX + 'px';
    particle.style.top = startY + 'px';
    
    topology.appendChild(particle);
    
    // Animate particle movement
    particle.animate([
        { left: startX + 'px', top: startY + 'px' },
        { left: endX + 'px', top: endY + 'px' }
    ], {
        duration: 2000,
        easing: 'ease-in-out'
    }).onfinish = () => {
        particle.remove();
    };
}

// Utility function
function adjustBrightness(hex, percent) {
    const num = parseInt(hex.replace("#",""), 16);
    const amt = Math.round(2.55 * percent);
    const R = (num >> 16) + amt;
    const G = (num >> 8 & 0x00FF) + amt;
    const B = (num & 0x0000FF) + amt;
    return "#" + (0x1000000 + (R<255?R<1?0:R:255)*0x10000 +
                  (G<255?G<1?0:G:255)*0x100 + (B<255?B<1?0:B:255))
                  .toString(16).slice(1);
}

// Enhanced fetch function for agent topology
async function fetchEnhancedAgentTopology() {
    try {
        const response = await fetch('/api/enhanced-agent-topology');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        
        updateEnhancedAgentTopology(data);
        
    } catch (error) {
        console.error('Error fetching enhanced agent topology:', error);
    }
}

// Enhanced workflow management
async function fetchWorkflows(page = 1) {
    try {
        const response = await fetch(`/api/workflows?page=${page}&page_size=10`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        workflowsData = await response.json();
        currentWorkflowPage = page;
        renderWorkflows();
        renderWorkflowPagination();
        updateWorkflowStats();
        
    } catch (error) {
        console.error('Error fetching workflows:', error);
        document.getElementById('workflows-container').innerHTML = 
            '<div style="text-align: center; color: #ef4444; padding: 40px;">Error loading workflows</div>';
    }
}

function renderWorkflows() {
    const container = document.getElementById('workflows-container');
    if (!container) return;
    
    if (workflowsData.workflows.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #9ca3af; padding: 40px;">No workflows found</div>';
        return;
    }
    
    container.innerHTML = workflowsData.workflows.map(workflow => {
        const statusClass = `status-${workflow.status}`;
        const duration = workflow.total_duration_seconds || 0;
        const durationText = duration > 0 ? `${duration.toFixed(1)}s` : 'In Progress';
        
        // ENHANCED: Display playbook information prominently
        const playbookDisplay = workflow.selected_playbook ? `
            <div class="playbook-info">
                <div class="playbook-name">📋 ${workflow.selected_playbook}</div>
                <div class="playbook-description">${workflow.playbook_description || 'Automated remediation playbook'}</div>
            </div>
        ` : '<div style="color: #9ca3af; font-style: italic; margin: 10px 0;">No playbook selected yet</div>';
        
        return `
            <div class="workflow-item">
                <div class="workflow-header">
                    <div class="workflow-id">${workflow.workflow_id}</div>
                    <div class="workflow-status ${statusClass}">${workflow.status.toUpperCase()}</div>
                </div>
                
                <div class="workflow-details">
                    <div class="workflow-detail-item">
                        <div class="workflow-detail-label">Component</div>
                        <div class="workflow-detail-value">${workflow.anomaly.component}</div>
                    </div>
                    <div class="workflow-detail-item">
                        <div class="workflow-detail-label">Severity</div>
                        <div class="workflow-detail-value">${workflow.anomaly.severity}</div>
                    </div>
                    <div class="workflow-detail-item">
                        <div class="workflow-detail-label">Duration</div>
                        <div class="workflow-detail-value">${durationText}</div>
                    </div>
                    <div class="workflow-detail-item">
                        <div class="workflow-detail-label">Created</div>
                        <div class="workflow-detail-value">${new Date(workflow.created_at).toLocaleString()}</div>
                    </div>
                    <div class="workflow-detail-item">
                        <div class="workflow-detail-label">Trigger Type</div>
                        <div class="workflow-detail-value">${workflow.anomaly.trigger_type.replace('_', ' ').toUpperCase()}</div>
                    </div>
                </div>
                
                ${playbookDisplay}
                
                <div class="workflow-steps">
                    <div style="font-weight: 600; margin-bottom: 10px; color: #f3f4f6;">Execution Steps:</div>
                    ${workflow.steps.map((step, index) => {
                        const stepStatusIcon = getStepStatusIcon(step.status);
                        const stepTiming = getStepTiming(step);
                        const isCurrentStep = index === workflow.current_step_index && workflow.status !== 'completed' && workflow.status !== 'failed';
                        
                        return `
                            <div class="workflow-step">
                                <div class="step-status-icon step-${step.status} ${isCurrentStep ? 'step-active' : ''}">${stepStatusIcon}</div>
                                <div class="step-details">
                                    <div>
                                        <span class="step-agent">${step.agent.toUpperCase()}</span>
                                        <span class="step-description">${step.description}</span>
                                    </div>
                                    <div class="step-timing">${stepTiming}</div>
                                    ${step.output ? `<div style="color: #10b981; font-size: 0.8em; margin-top: 3px;">✓ ${step.output}</div>` : ''}
                                    ${step.error ? `<div style="color: #ef4444; font-size: 0.8em; margin-top: 3px;">✗ ${step.error}</div>` : ''}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }).join('');
}

function getStepStatusIcon(status) {
    switch (status) {
        case 'completed': return '✓';
        case 'active': return '⚡';
        case 'failed': return '✗';
        case 'pending': return '○';
        default: return '○';
    }
}

function getStepTiming(step) {
    if (step.completed_at && step.started_at) {
        const duration = step.duration_seconds;
        return `Duration: ${duration.toFixed(1)}s (${new Date(step.started_at).toLocaleTimeString()} - ${new Date(step.completed_at).toLocaleTimeString()})`;
    } else if (step.started_at) {
        return `Started: ${new Date(step.started_at).toLocaleTimeString()}`;
    } else {
        return 'Pending';
    }
}

function renderWorkflowPagination() {
    const container = document.getElementById('workflows-pagination');
    if (!container || !workflowsData.pagination) return;
    
    const p = workflowsData.pagination;
    
    container.innerHTML = `
        <button class="pagination-button" ${!p.has_previous ? 'disabled' : ''} onclick="fetchWorkflows(${p.current_page - 1})">
            Previous
        </button>
        <span class="pagination-info">
            Page ${p.current_page} of ${p.total_pages} (${p.total_workflows} total workflows)
        </span>
        <button class="pagination-button" ${!p.has_next ? 'disabled' : ''} onclick="fetchWorkflows(${p.current_page + 1})">
            Next
        </button>
    `;
}

function updateWorkflowStats() {
    if (workflowsData.stats) {
        const activeCount = document.getElementById('active-workflows-count');
        const completedCount = document.getElementById('completed-workflows-count');
        const totalCount = document.getElementById('total-workflows-count');
        
        if (activeCount) activeCount.textContent = workflowsData.stats.active_workflows;
        if (completedCount) completedCount.textContent = workflowsData.stats.completed_workflows;
        if (totalCount) totalCount.textContent = workflowsData.stats.total_workflows;
    }
}

// Enhanced progress fetching
async function fetchProgress() {
    try {
        const response = await fetch('/api/progress');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        progressData = await response.json();
        
        updateEnhancedTimeBar();
        
        // Fetch metrics data for charts
        await fetchMetrics();
        
    } catch (error) {
        console.error('Error fetching progress:', error);
    }
}

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all nav tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked nav tab
    event.target.classList.add('active');
    
    // Load tab-specific content
    if (tabName === 'workflows') {
        fetchWorkflows(1);
    } else if (tabName === 'agents') {
        fetchEnhancedAgentTopology();
    }
}

// FIXED: Fetch system stats with discovered counts
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const stats = await response.json();
        
        // Use discovered_anomaly_counts instead of anomaly_counts
        document.getElementById('amf-count').textContent = stats.discovered_anomaly_counts?.AMF || 0;
        document.getElementById('smf-count').textContent = stats.discovered_anomaly_counts?.SMF || 0;
        document.getElementById('upf-count').textContent = stats.discovered_anomaly_counts?.UPF || 0;
        document.getElementById('processed-count').textContent = stats.anomalies_processed || 0;
        document.getElementById('executed-count').textContent = stats.playbooks_executed || 0;
        
        const successRate = stats.playbooks_executed > 0 ? 
            Math.round((stats.successful_executions / stats.playbooks_executed) * 100) : 100;
        document.getElementById('success-rate').textContent = successRate;
        
        // Add anomaly markers to charts
        if (stats.recent_anomalies) {
            stats.recent_anomalies.forEach(anomaly => {
                addAnomalyToChart(anomaly.component, anomaly.timestamp, anomaly.severity);
            });
        }
        
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

// Fetch metrics data
async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        
        // Update metrics data
        updateMetricsFromAPI(data);
        
    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
}

// Update metrics from API data
function updateMetricsFromAPI(data) {
    if (!data) return;
    
    // AMF Metrics
    if (amfChart && data.amf) {
        const amfData = data.amf;
        if (amfData.timestamps && amfData.metrics) {
            amfChart.data.labels = amfData.timestamps.map(t => new Date(t).toLocaleTimeString());
            amfChart.data.datasets[0].data = amfData.metrics.registration_rate || [];
            amfChart.data.datasets[1].data = amfData.metrics.registration_success_rate || [];
            amfChart.data.datasets[2].data = amfData.metrics.auth_success_rate || [];
            amfChart.update('none');
            
            // Update anomalies
            if (amfData.anomalies) {
                metricsData.amf.anomalies = amfData.anomalies;
            }
        }
    }
    
    // SMF Metrics
    if (smfChart && data.smf) {
        const smfData = data.smf;
        if (smfData.timestamps && smfData.metrics) {
            smfChart.data.labels = smfData.timestamps.map(t => new Date(t).toLocaleTimeString());
            smfChart.data.datasets[0].data = smfData.metrics.session_est_rate || [];
            smfChart.data.datasets[1].data = smfData.metrics.session_success_rate || [];
            smfChart.data.datasets[2].data = smfData.metrics.ip_pool_usage || [];
            smfChart.update('none');
            
            if (smfData.anomalies) {
                metricsData.smf.anomalies = smfData.anomalies;
            }
        }
    }
    
    // UPF Metrics
    if (upfChart && data.upf) {
        const upfData = data.upf;
        if (upfData.timestamps && upfData.metrics) {
            upfChart.data.labels = upfData.timestamps.map(t => new Date(t).toLocaleTimeString());
            upfChart.data.datasets[0].data = upfData.metrics.active_sessions || [];
            upfChart.data.datasets[1].data = upfData.metrics.throughput || [];
            upfChart.data.datasets[2].data = upfData.metrics.latency || [];
            upfChart.update('none');
            
            if (upfData.anomalies) {
                metricsData.upf.anomalies = upfData.anomalies;
            }
        }
    }
}

// Trigger event
async function triggerEvent(component, severity) {
    try {
        const response = await fetch(`/api/trigger/${component}?severity=${severity}`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        
        // Refresh data after triggering
        setTimeout(() => {
            fetchStats();
            fetchProgress();
            if (document.getElementById('workflows-tab').classList.contains('active')) {
                fetchWorkflows(currentWorkflowPage);
            }
        }, 2000);
        
    } catch (error) {
        console.error('Error triggering event:', error);
    }
}

// Initialize dashboard
function initDashboard() {
    initEnhancedAgentTopology();
    initMetricCharts();
    fetchStats();
    fetchProgress();
    fetchWorkflows(1);
    fetchEnhancedAgentTopology();
    fetchMetrics(); // Initial metrics fetch
    
    // Auto-refresh data
    setInterval(() => {
        fetchStats();
        fetchProgress();
        fetchEnhancedAgentTopology();
        
        // Refresh workflows if tab is active
        if (document.getElementById('workflows-tab').classList.contains('active')) {
            fetchWorkflows(currentWorkflowPage);
        }
    }, 3000);
}

// Start dashboard when page loads
document.addEventListener('DOMContentLoaded', initDashboard);