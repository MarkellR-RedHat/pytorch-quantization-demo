// Audience interface JavaScript

let selectedModel = null;
let yourRequestCount = 0;
let ws = null;

// DOM Elements
const modelButtons = document.querySelectorAll('.model-btn');
const sendButton = document.getElementById('sendRequestBtn');
const yourRequestsEl = document.getElementById('yourRequests');
const totalRequestsEl = document.getElementById('totalRequests');
const participantsEl = document.getElementById('participants');
const statusMessageEl = document.getElementById('statusMessage');

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        showStatus('Connected! Ready to send requests.', 'success');
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showStatus('Connection error. Retrying...', 'error');
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        showStatus('Disconnected. Reconnecting...', 'warning');
        setTimeout(initWebSocket, 3000);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'metrics_update':
            updateMetrics(message.data);
            break;
        case 'state_update':
            updateState(message.data);
            break;
    }
}

// Update metrics display
function updateMetrics(metrics) {
    // Calculate total requests across all models
    let total = 0;
    for (const model in metrics) {
        total += metrics[model].total_requests;
    }
    totalRequestsEl.textContent = total;
}

// Update demo state
function updateState(state) {
    participantsEl.textContent = state.participant_count;
    totalRequestsEl.textContent = state.total_requests;
}

// Model selection
modelButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Remove selection from all buttons
        modelButtons.forEach(btn => btn.classList.remove('selected'));

        // Select this button
        button.classList.add('selected');
        selectedModel = button.dataset.model;

        // Enable send button
        sendButton.disabled = false;
        sendButton.querySelector('.btn-subtext').textContent = `Send to ${selectedModel}`;

        showStatus(`${selectedModel} selected. Ready to send!`, 'success');
    });
});

// Send request
sendButton.addEventListener('click', async () => {
    if (!selectedModel) {
        showStatus('Please select a model first', 'warning');
        return;
    }

    // Disable button temporarily
    sendButton.disabled = true;
    showStatus('Sending request...', 'info');

    try {
        const response = await fetch('/request', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_type: selectedModel,
                prompt: 'Hello, how are you?'
            })
        });

        if (!response.ok) {
            throw new Error('Request failed');
        }

        const data = await response.json();

        // Increment your request count
        yourRequestCount++;
        yourRequestsEl.textContent = yourRequestCount;

        // Show success
        showStatus(`Request sent! Latency: ${Math.round(data.latency_ms)}ms`, 'success');

    } catch (error) {
        console.error('Error sending request:', error);
        showStatus('Failed to send request. Try again.', 'error');
    } finally {
        // Re-enable button after short delay
        setTimeout(() => {
            sendButton.disabled = false;
        }, 500);
    }
});

// Show status message
function showStatus(message, type = 'info') {
    statusMessageEl.textContent = message;
    statusMessageEl.className = `status-message ${type}`;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    showStatus('Select a model to get started', 'info');
});
