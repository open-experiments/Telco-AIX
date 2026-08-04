// agentic/dashboard/static/js/agent.js
document.addEventListener('DOMContentLoaded', function() {
    // Get agent ID from URL
    const pathParts = window.location.pathname.split('/');
    const agentId = pathParts[pathParts.length - 1];
    
    // Initialize WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() {
        console.log('WebSocket connection established');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data, agentId);
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
    
    // Fetch agent data
    fetchAgentData(agentId);
    initAgentActivityChart();
});

// WebSocket message handler
function handleWebSocketMessage(data, agentId) {
    if (data.event_type === 'agents_update') {
        // Find the agent in the update
        const agent = data.data.find(a => a.agent_id === agentId);
        if (agent) {
            updateAgentDetails(agent);
        }
    } else if (data.event_type === 'agent_message' && data.data.agent_id === agentId) {
        addAgentMessage(data.data.message);
    }
}

// Fetch agent data
function fetchAgentData(agentId) {
    fetch(`/api/agents/${agentId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Agent not found: ${agentId}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                updateAgentDetails(data.agent);
                fetchAgentMessages(agentId);
            }
        })
        .catch(error => {
            console.error('Error fetching agent data:', error);
            document.getElementById('agentName').textContent = 'Agent not found';
        });
}

// Fetch agent messages
function fetchAgentMessages(agentId) {
    fetch(`/api/agents/${agentId}/messages`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateAgentMessages(data.messages);
            }
        })
        .catch(error => console.error('Error fetching agent messages:', error));
}

// Update agent details
function updateAgentDetails(agent) {
    document.getElementById('agentName').textContent = agent.name;
    document.getElementById('agentBreadcrumb').textContent = agent.name;
    document.getElementById('agentType').textContent = agent.agent_type;
    document.getElementById('agentStatus').textContent = agent.status;
    
    document.getElementById('agentId').textContent = agent.agent_id;
    document.getElementById('agentTypeInfo').textContent = agent.agent_type;
    document.getElementById('agentStatusInfo').textContent = agent.status;
    document.getElementById('agentLastSeen').textContent = formatDateTime(agent.last_seen);
    
    // Format network location
    const location = agent.network_location;
    document.getElementById('agentLocation').textContent = `${location.host}:${location.port}`;
    
    // Update capabilities
    const capabilitiesContainer = document.getElementById('agentCapabilities');
    capabilitiesContainer.innerHTML = '';
    
    const capabilities = agent.capabilities;
    
    // Add action types
    capabilities.action_types.forEach(action => {
        const capabilityItem = document.createElement('div');
        capabilityItem.className = 'capability-item';
        capabilityItem.textContent = `Action: ${action}`;
        capabilitiesContainer.appendChild(capabilityItem);
    });
    
    // Add domains
    capabilities.domains.forEach(domain => {
        const capabilityItem = document.createElement('div');
        capabilityItem.className = 'capability-item';
        capabilityItem.textContent = `Domain: ${domain}`;
        capabilitiesContainer.appendChild(capabilityItem);
    });
    
    // Add description
    const descriptionItem = document.createElement('div');
    descriptionItem.className = 'capability-item';
    descriptionItem.textContent = capabilities.description;
    capabilitiesContainer.appendChild(descriptionItem);
}

// Update agent messages
function updateAgentMessages(messages) {
    const messagesContainer = document.getElementById('agentMessages');
    messagesContainer.innerHTML = '';
    
    if (!messages || messages.length === 0) {
        const noMessages = document.createElement('div');
        noMessages.className = 'message';
        noMessages.textContent = 'No messages to display';
        messagesContainer.appendChild(noMessages);
        return;
    }
    
    messages.forEach(message => {
        addAgentMessage(message);
    });
}

// Add a single agent message
function addAgentMessage(message) {
    const messagesContainer = document.getElementById('agentMessages');
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message';

    const time = new Date(message.timestamp);

    // Build elements with textContent (never innerHTML) to avoid XSS
    const header = document.createElement('div');
    header.className = 'message-header';

    const typeSpan = document.createElement('span');
    typeSpan.className = `message-type ${message.type.toLowerCase()}`;
    typeSpan.textContent = message.type;

    const timeSpan = document.createElement('span');
    timeSpan.className = 'message-time';
    timeSpan.textContent = formatDateTime(time);

    header.appendChild(typeSpan);
    header.appendChild(timeSpan);

    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = message.content;

    messageElement.appendChild(header);
    messageElement.appendChild(content);
    
    messagesContainer.insertBefore(messageElement, messagesContainer.firstChild);
}

// Initialize agent activity chart
function initAgentActivityChart() {
    const activityCtx = document.getElementById('agentActivityChart').getContext('2d');
    window.agentActivityChart = new Chart(activityCtx, {
        type: 'bar',
        data: {
            labels: generateTimeLabels(24), // Last 24 hours
            datasets: [{
                label: 'Actions Performed',
                data: generateRandomData(24, 0, 10),
                backgroundColor: '#1e88e5'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
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
        const time = new Date(now - i * 3600000); // 1 hour ago
        labels.push(formatTime(time));
    }
    
    return labels;
}

function generateRandomData(count, min, max) {
    return Array.from({length: count}, () => Math.floor(Math.random() * (max - min + 1)) + min);
}
