// agentic/dashboard/static/js/dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navLinks = document.querySelectorAll('nav ul li a');
    const sections = document.querySelectorAll('.section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            
            // Hide all sections
            sections.forEach(section => {
                section.classList.remove('active');
            });
            
            // Show target section
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add('active');
            }
            
            // Update active nav link
            navLinks.forEach(navLink => {
                navLink.parentElement.classList.remove('active');
            });
            this.parentElement.classList.add('active');
        });
    });
    
    // Initialize WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() {
        console.log('WebSocket connection established');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = function() {
        console.log('WebSocket connection closed');
        // Try to reconnect after a few seconds
        setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            initWebSocket();
        }, 5000);
    };
    
    // Fetch initial data
    fetchAgents();
    fetchWorkflows();
    initCharts();
    
    // Add event listeners for filters
    document.getElementById('agentTypeFilter').addEventListener('change', filterAgents);
    document.getElementById('agentStatusFilter').addEventListener('change', filterAgents);
    document.getElementById('agentSearch').addEventListener('input', filterAgents);
    
    document.getElementById('workflowTypeFilter').addEventListener('change', filterWorkflows);
    document.getElementById('workflowStatusFilter').addEventListener('change', filterWorkflows);
    document.getElementById('workflowSearch').addEventListener('input', filterWorkflows);
});

// WebSocket message handler
function handleWebSocketMessage(data) {
    if (data.event_type === 'agents_update') {
        updateAgentsTable(data.data);
        updateAgentStatusChart(data.data);
    } else if (data.event_type === 'workflows_update') {
        updateWorkflowsTable(data.data);
        updateWorkflowStatusChart(data.data);
    } else if (data.event_type === 'telemetry_update') {
        updateTelemetryCharts(data.data);
    } else if (data.event_type === 'event') {
        addEvent(data.data);
    }
}

// Fetch agents from API
function fetchAgents() {
    fetch('/api/agents')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateAgentsTable(data.agents);
                updateAgentStatusChart(data.agents);
            }
        })
        .catch(error => console.error('Error fetching agents:', error));
}

// Fetch workflows from API
function fetchWorkflows() {
    fetch('/api/workflows')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateWorkflowsTable(Object.values(data.workflows));
                updateWorkflowStatusChart(Object.values(data.workflows));
            }
        })
        .catch(error => console.error('Error fetching workflows:', error));
}

// Helper: create a <td> with safe text content (prevents XSS)
function createCell(text) {
    const td = document.createElement('td');
    td.textContent = text == null ? '' : String(text);
    return td;
}

// Update agents table
function updateAgentsTable(agents) {
    const tableBody = document.getElementById('agentsTableBody');
    tableBody.innerHTML = '';

    agents.forEach(agent => {
        const row = document.createElement('tr');

        const statusClass = `status-${agent.status.toLowerCase()}`;

        // Build cells with textContent (never innerHTML) to avoid XSS
        row.appendChild(createCell(agent.agent_id));
        row.appendChild(createCell(agent.name));
        row.appendChild(createCell(agent.agent_type));

        const statusTd = document.createElement('td');
        const statusBadge = document.createElement('span');
        statusBadge.className = `status-badge ${statusClass}`;
        statusBadge.textContent = agent.status;
        statusTd.appendChild(statusBadge);
        row.appendChild(statusTd);

        row.appendChild(createCell(formatDateTime(agent.last_seen)));

        const actionTd = document.createElement('td');
        const detailsLink = document.createElement('a');
        detailsLink.href = `/agents/${encodeURIComponent(agent.agent_id)}`;
        detailsLink.className = 'action-button';
        detailsLink.textContent = 'Details';
        actionTd.appendChild(detailsLink);
        row.appendChild(actionTd);

        tableBody.appendChild(row);
    });

    filterAgents();
}

// Update workflows table
function updateWorkflowsTable(workflows) {
    const tableBody = document.getElementById('workflowsTableBody');
    tableBody.innerHTML = '';
    
    workflows.forEach(workflow => {
        const row = document.createElement('tr');

        const statusClass = `status-${workflow.status.toLowerCase()}`;
        const progress = calculateWorkflowProgress(workflow);

        // Build cells with textContent (never innerHTML) to avoid XSS
        row.appendChild(createCell(workflow.workflow_id));
        row.appendChild(createCell(workflow.workflow_type));

        const statusTd = document.createElement('td');
        const statusBadge = document.createElement('span');
        statusBadge.className = `status-badge ${statusClass}`;
        statusBadge.textContent = workflow.status;
        statusTd.appendChild(statusBadge);
        row.appendChild(statusTd);

        row.appendChild(createCell(formatDateTime(workflow.created_at)));

        const progressTd = document.createElement('td');
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        const progressFill = document.createElement('div');
        progressFill.className = 'progress';
        progressFill.style.width = `${Number(progress) || 0}%`;
        progressBar.appendChild(progressFill);
        progressTd.appendChild(progressBar);
        row.appendChild(progressTd);

        const actionTd = document.createElement('td');
        const detailsLink = document.createElement('a');
        detailsLink.href = `/workflows/${encodeURIComponent(workflow.workflow_id)}`;
        detailsLink.className = 'action-button';
        detailsLink.textContent = 'Details';
        actionTd.appendChild(detailsLink);
        row.appendChild(actionTd);

        tableBody.appendChild(row);
    });
    
    filterWorkflows();
}

// Filter agents based on selected filters
function filterAgents() {
    const typeFilter = document.getElementById('agentTypeFilter').value;
    const statusFilter = document.getElementById('agentStatusFilter').value;
    const searchValue = document.getElementById('agentSearch').value.toLowerCase();
    
    const rows = document.getElementById('agentsTableBody').getElementsByTagName('tr');
    
    for (let i = 0; i < rows.length; i++) {
        const typeCell = rows[i].getElementsByTagName('td')[2];
        const statusCell = rows[i].getElementsByTagName('td')[3];
        const nameCell = rows[i].getElementsByTagName('td')[1];
        const idCell = rows[i].getElementsByTagName('td')[0];
        
        const type = typeCell.textContent;
        const status = statusCell.textContent;
        const name = nameCell.textContent.toLowerCase();
        const id = idCell.textContent.toLowerCase();
        
        const typeMatch = typeFilter === 'all' || type === typeFilter;
        const statusMatch = statusFilter === 'all' || status === statusFilter;
        const searchMatch = name.includes(searchValue) || id.includes(searchValue);
        
        if (typeMatch && statusMatch && searchMatch) {
            rows[i].style.display = '';
        } else {
            rows[i].style.display = 'none';
        }
    }
}

// Filter workflows based on selected filters
function filterWorkflows() {
    const typeFilter = document.getElementById('workflowTypeFilter').value;
    const statusFilter = document.getElementById('workflowStatusFilter').value;
    const searchValue = document.getElementById('workflowSearch').value.toLowerCase();
    
    const rows = document.getElementById('workflowsTableBody').getElementsByTagName('tr');
    
    for (let i = 0; i < rows.length; i++) {
        const typeCell = rows[i].getElementsByTagName('td')[1];
        const statusCell = rows[i].getElementsByTagName('td')[2];
        const idCell = rows[i].getElementsByTagName('td')[0];
        
        const type = typeCell.textContent;
        const status = statusCell.textContent;
        const id = idCell.textContent.toLowerCase();
        
        const typeMatch = typeFilter === 'all' || type === typeFilter;
        const statusMatch = statusFilter === 'all' || status === statusFilter;
        const searchMatch = id.includes(searchValue);
        
        if (typeMatch && statusMatch && searchMatch) {
            rows[i].style.display = '';
        } else {
            rows[i].style.display = 'none';
        }
    }
}

// Initialize charts
function initCharts() {
    // Agent Status Chart
    const agentStatusCtx = document.getElementById('agentStatusChart').getContext('2d');
    window.agentStatusChart = new Chart(agentStatusCtx, {
        type: 'doughnut',
        data: {
            labels: ['Active', 'Idle', 'Processing', 'Error'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: [
                    '#4caf50',
                    '#2196f3',
                    '#ff9800',
                    '#f44336'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
    
    // Workflow Status Chart
    const workflowStatusCtx = document.getElementById('workflowStatusChart').getContext('2d');
    window.workflowStatusChart = new Chart(workflowStatusCtx, {
        type: 'doughnut',
        data: {
            labels: ['Created', 'Running', 'Completed', 'Failed'],
            datasets: [{
                data: [0, 0, 0, 0],
                backgroundColor: [
                    '#2196f3',
                    '#ff9800',
                    '#4caf50',
                    '#f44336'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
    
    // Telemetry Charts
    const packetLossCtx = document.getElementById('packetLossChart').getContext('2d');
    window.packetLossChart = new Chart(packetLossCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'Packet Loss %',
                data: generateRandomData(20, 0, 2),
                borderColor: '#f44336',
                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 5
                }
            }
        }
    });
    
    const latencyCtx = document.getElementById('latencyChart').getContext('2d');
    window.latencyChart = new Chart(latencyCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'Latency (ms)',
                data: generateRandomData(20, 30, 80),
                borderColor: '#ff9800',
                backgroundColor: 'rgba(255, 152, 0, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    const throughputCtx = document.getElementById('throughputChart').getContext('2d');
    window.throughputChart = new Chart(throughputCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'Throughput (Mbps)',
                data: generateRandomData(20, 800, 1200),
                borderColor: '#4caf50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    const signalStrengthCtx = document.getElementById('signalStrengthChart').getContext('2d');
    window.signalStrengthChart = new Chart(signalStrengthCtx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(20),
            datasets: [{
                label: 'Signal Strength (dBm)',
                data: generateRandomData(20, -80, -60),
                borderColor: '#2196f3',
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Update agent status chart
function updateAgentStatusChart(agents) {
    const statusCounts = {
        'active': 0,
        'idle': 0,
        'processing': 0,
        'error': 0
    };
    
    agents.forEach(agent => {
        const status = agent.status.toLowerCase();
        if (status in statusCounts) {
            statusCounts[status]++;
        }
    });
    
    window.agentStatusChart.data.datasets[0].data = [
        statusCounts.active,
        statusCounts.idle,
        statusCounts.processing,
        statusCounts.error
    ];
    
    window.agentStatusChart.update();
}

// Update workflow status chart
function updateWorkflowStatusChart(workflows) {
    const statusCounts = {
        'created': 0,
        'running': 0,
        'completed': 0,
        'failed': 0
    };
    
    workflows.forEach(workflow => {
        const status = workflow.status.toLowerCase();
        if (status in statusCounts) {
            statusCounts[status]++;
        }
    });
    
    window.workflowStatusChart.data.datasets[0].data = [
        statusCounts.created,
        statusCounts.running,
        statusCounts.completed,
        statusCounts.failed
    ];
    
    window.workflowStatusChart.update();
}

// Update telemetry charts
function updateTelemetryCharts(telemetry) {
    // Update packet loss chart
    if (telemetry.packet_loss !== undefined) {
        window.packetLossChart.data.labels.shift();
        window.packetLossChart.data.labels.push(formatTime(new Date()));
        window.packetLossChart.data.datasets[0].data.shift();
        window.packetLossChart.data.datasets[0].data.push(telemetry.packet_loss * 100); // Convert to percentage
        window.packetLossChart.update();
    }
    
    // Update latency chart
    if (telemetry.latency !== undefined) {
        window.latencyChart.data.labels.shift();
        window.latencyChart.data.labels.push(formatTime(new Date()));
        window.latencyChart.data.datasets[0].data.shift();
        window.latencyChart.data.datasets[0].data.push(telemetry.latency);
        window.latencyChart.update();
    }
    
    // Update throughput chart
    if (telemetry.throughput !== undefined) {
        window.throughputChart.data.labels.shift();
        window.throughputChart.data.labels.push(formatTime(new Date()));
        window.throughputChart.data.datasets[0].data.shift();
        window.throughputChart.data.datasets[0].data.push(telemetry.throughput);
        window.throughputChart.update();
    }
    
    // Update signal strength chart
    if (telemetry.signal_strength !== undefined) {
        window.signalStrengthChart.data.labels.shift();
        window.signalStrengthChart.data.labels.push(formatTime(new Date()));
        window.signalStrengthChart.data.datasets[0].data.shift();
        window.signalStrengthChart.data.datasets[0].data.push(telemetry.signal_strength);
        window.signalStrengthChart.update();
    }
}

// Add a new event to the events list
function addEvent(event) {
    const eventsContainer = document.getElementById('eventsContainer');
    
    const eventElement = document.createElement('div');
    eventElement.className = 'event';

    const time = new Date(event.timestamp);

    // Build spans with textContent (never innerHTML) to avoid XSS
    const timeSpan = document.createElement('span');
    timeSpan.className = 'event-time';
    timeSpan.textContent = formatTime(time);

    const typeSpan = document.createElement('span');
    typeSpan.className = `event-type ${event.agent_type}`;
    typeSpan.textContent = event.agent_type;

    const messageSpan = document.createElement('span');
    messageSpan.className = 'event-message';
    messageSpan.textContent = event.message;

    eventElement.appendChild(timeSpan);
    eventElement.appendChild(typeSpan);
    eventElement.appendChild(messageSpan);
    
    eventsContainer.insertBefore(eventElement, eventsContainer.firstChild);
    
    // Limit the number of events shown
    if (eventsContainer.children.length > 10) {
        eventsContainer.removeChild(eventsContainer.lastChild);
    }
}

// Calculate workflow progress percentage
function calculateWorkflowProgress(workflow) {
    if (workflow.status === 'completed') {
        return 100;
    }
    
    if (workflow.steps_total === 0) {
        return 0;
    }
    
    return Math.round((workflow.steps_completed / workflow.steps_total) * 100);
}

// Helper functions
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return `${date.toLocaleDateString()} ${formatTime(date)}`;
}

function formatTime(date) {
    return date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
}

function generateTimeLabels(count) {
    const labels = [];
    const now = new Date();
    
    for (let i = count - 1; i >= 0; i--) {
        const time = new Date(now - i * 30000); // 30 seconds ago
        labels.push(formatTime(time));
    }
    
    return labels;
}

function generateRandomData(count, min, max) {
    return Array.from({length: count}, () => Math.random() * (max - min) + min);
}
