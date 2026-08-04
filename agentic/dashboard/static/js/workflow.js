// agentic/dashboard/static/js/workflow.js
document.addEventListener('DOMContentLoaded', function() {
    // Get workflow ID from URL
    const pathParts = window.location.pathname.split('/');
    const workflowId = pathParts[pathParts.length - 1];
    
    // Initialize WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() {
        console.log('WebSocket connection established');
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data, workflowId);
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
    
    // Fetch workflow data
    fetchWorkflowData(workflowId);
});

// WebSocket message handler
function handleWebSocketMessage(data, workflowId) {
    if (data.event_type === 'workflows_update') {
        // Find the workflow in the update
        const workflow = data.data.find(w => w.workflow_id === workflowId);
        if (workflow) {
            updateWorkflowDetails(workflow);
        }
    }
}

// Fetch workflow data
function fetchWorkflowData(workflowId) {
    fetch(`/api/workflows/${workflowId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Workflow not found: ${workflowId}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                updateWorkflowDetails(data.workflow);
            }
        })
        .catch(error => {
            console.error('Error fetching workflow data:', error);
            document.getElementById('workflowTitle').textContent = 'Workflow not found';
        });
}

// Update workflow details
function updateWorkflowDetails(workflow) {
    document.getElementById('workflowTitle').textContent = `${workflow.workflow_type} Workflow`;
    document.getElementById('workflowBreadcrumb').textContent = `${workflow.workflow_type} Workflow`;
    document.getElementById('workflowType').textContent = workflow.workflow_type;
    
    const statusClass = `status-${workflow.status.toLowerCase()}`;
    const statusElement = document.getElementById('workflowStatus');
    statusElement.textContent = workflow.status;
    statusElement.className = `status-badge ${statusClass}`;
    
    document.getElementById('workflowId').textContent = workflow.workflow_id;
    document.getElementById('workflowTypeInfo').textContent = workflow.workflow_type;
    document.getElementById('workflowStatusInfo').textContent = workflow.status;
    document.getElementById('workflowCreated').textContent = formatDateTime(workflow.created_at);
    document.getElementById('workflowUpdated').textContent = formatDateTime(workflow.updated_at);
    document.getElementById('workflowInitiator').textContent = workflow.initiator_id || 'System';
    
    // Update progress
    const progress = calculateWorkflowProgress(workflow);
    document.getElementById('workflowProgress').style.width = `${progress}%`;
    document.getElementById('stepsCompleted').textContent = workflow.steps_completed || 0;
    document.getElementById('stepsTotal').textContent = workflow.steps_total || 0;
    
    // Update parameters
    const parametersContainer = document.getElementById('workflowParameters');
    parametersContainer.innerHTML = '';
    
    if (!workflow.parameters || Object.keys(workflow.parameters).length === 0) {
        parametersContainer.textContent = 'No parameters defined';
    } else {
        Object.entries(workflow.parameters).forEach(([key, value]) => {
            const paramItem = document.createElement('div');
            paramItem.className = 'parameter-item';
            // Build spans with textContent (never innerHTML) to avoid XSS
            const label = document.createElement('span');
            label.className = 'param-label';
            label.textContent = `${key}:`;
            const val = document.createElement('span');
            val.className = 'param-value';
            val.textContent = JSON.stringify(value);
            paramItem.appendChild(label);
            paramItem.appendChild(val);
            parametersContainer.appendChild(paramItem);
        });
    }
    
    // Update results
    const resultsContainer = document.getElementById('workflowResults');
    resultsContainer.innerHTML = '';
    
    if (!workflow.results || Object.keys(workflow.results).length === 0) {
        resultsContainer.textContent = 'No results available';
    } else {
        Object.entries(workflow.results).forEach(([key, value]) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';
            // Build spans with textContent (never innerHTML) to avoid XSS
            const label = document.createElement('span');
            label.className = 'result-label';
            label.textContent = `${key}:`;
            const val = document.createElement('span');
            val.className = 'result-value';
            val.textContent = JSON.stringify(value);
            resultItem.appendChild(label);
            resultItem.appendChild(val);
            resultsContainer.appendChild(resultItem);
        });
    }
    
    // Update steps
    const stepsContainer = document.getElementById('workflowSteps');
    stepsContainer.innerHTML = '';
    
    if (!workflow.steps || workflow.steps.length === 0) {
        const noSteps = document.createElement('div');
        noSteps.className = 'step';
        noSteps.textContent = 'No steps defined for this workflow';
        stepsContainer.appendChild(noSteps);
    } else {
        workflow.steps.forEach(step => {
            const stepElement = document.createElement('div');
            stepElement.className = `step ${getStepStatusClass(step.status)}`;

            // Build elements with textContent (never innerHTML) to avoid XSS
            const header = document.createElement('div');
            header.className = 'step-header';

            const titleSpan = document.createElement('span');
            titleSpan.className = 'step-title';
            titleSpan.textContent = step.step_id;

            const statusSpan = document.createElement('span');
            statusSpan.className = `step-status ${getStepStatusClass(step.status)}`;
            statusSpan.textContent = step.status;

            header.appendChild(titleSpan);
            header.appendChild(statusSpan);

            const content = document.createElement('div');
            content.className = 'step-content';
            content.textContent = step.description || 'No description available';

            stepElement.appendChild(header);
            stepElement.appendChild(content);

            stepsContainer.appendChild(stepElement);
        });
    }
}

// Helper functions
function calculateWorkflowProgress(workflow) {
    if (workflow.status === 'completed') {
        return 100;
    }
    
    if (!workflow.steps_total || workflow.steps_total === 0) {
        return 0;
    }
    
    return Math.round((workflow.steps_completed / workflow.steps_total) * 100);
}

function getStepStatusClass(status) {
    status = status ? status.toLowerCase() : '';
    
    if (status === 'completed') {
        return 'completed';
    } else if (status === 'in_progress' || status === 'running') {
        return 'in-progress';
    } else if (status === 'failed' || status === 'error') {
        return 'failed';
    }
    
    return '';
}

function formatDateTime(dateString) {
    try {
        const date = new Date(dateString);
        return `${date.toLocaleDateString()} ${formatTime(date)}`;
    } catch (e) {
        return dateString || 'N/A';
    }
}

function formatTime(date) {
    return date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
}
