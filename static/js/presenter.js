// Presenter dashboard JavaScript

let ws = null;
let simulationMode = false;

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/presenter`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('Presenter WebSocket connected');
        fetchInitialState();
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected. Reconnecting...');
        setTimeout(initWebSocket, 3000);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'metrics_update':
            updateMetricsDisplay(message.data);
            break;
        case 'state_update':
            updateStateDisplay(message.data);
            break;
        case 'simulation_toggled':
            simulationMode = message.data.enabled;
            updateModeIndicator();
            break;
    }
}

// Fetch initial state
async function fetchInitialState() {
    try {
        const [metricsRes, stateRes] = await Promise.all([
            fetch('/metrics'),
            fetch('/state')
        ]);

        const metrics = await metricsRes.json();
        const state = await stateRes.json();

        updateMetricsDisplay(metrics);
        updateStateDisplay(state);
        simulationMode = state.simulation_mode;
        updateModeIndicator();

    } catch (error) {
        console.error('Error fetching initial state:', error);
    }
}

// Update metrics display
function updateMetricsDisplay(metrics) {
    for (const [modelType, modelMetrics] of Object.entries(metrics)) {
        const card = document.querySelector(`.metric-card[data-model="${modelType}"]`);
        if (!card) continue;

        // Update each metric value
        card.querySelector('[data-metric="rps"]').textContent = modelMetrics.requests_per_second.toFixed(1);
        card.querySelector('[data-metric="latency"]').textContent = `${Math.round(modelMetrics.avg_latency_ms)} ms`;
        card.querySelector('[data-metric="p95"]').textContent = `${Math.round(modelMetrics.p95_latency_ms)} ms`;
        card.querySelector('[data-metric="memory"]').textContent = `${modelMetrics.gpu_memory_gb.toFixed(1)} GB`;
        card.querySelector('[data-metric="tps"]').textContent = modelMetrics.tokens_per_second.toFixed(1);
        card.querySelector('[data-metric="cost"]').textContent = `$${modelMetrics.cost_per_request.toFixed(4)}`;
        card.querySelector('[data-metric="total"]').textContent = modelMetrics.total_requests;
    }
}

// Update state display
function updateStateDisplay(state) {
    document.getElementById('totalParticipants').textContent = state.participant_count;
    document.getElementById('totalDemoRequests').textContent = state.total_requests;
}

// Update mode indicator
function updateModeIndicator() {
    const indicator = document.getElementById('modeIndicator');
    if (simulationMode) {
        indicator.textContent = 'SIMULATION MODE';
        indicator.classList.add('simulation');
    } else {
        indicator.textContent = 'LIVE MODE';
        indicator.classList.remove('simulation');
    }
}

// Toggle simulation mode
async function toggleSimulation() {
    try {
        const response = await fetch('/simulation/toggle', {
            method: 'POST'
        });

        const data = await response.json();
        console.log(data.message);

    } catch (error) {
        console.error('Error toggling simulation:', error);
    }
}

// Reset demo
async function resetDemo() {
    if (!confirm('Are you sure you want to reset the demo? This will clear all metrics.')) {
        return;
    }

    try {
        const response = await fetch('/demo/reset', {
            method: 'POST'
        });

        const data = await response.json();
        console.log(data.message);

    } catch (error) {
        console.error('Error resetting demo:', error);
    }
}

// Event listeners
document.getElementById('toggleSimBtn').addEventListener('click', toggleSimulation);
document.getElementById('resetBtn').addEventListener('click', resetDemo);

// Keyboard shortcut for simulation mode toggle
document.addEventListener('keydown', (event) => {
    // Ctrl+Shift+S
    if (event.ctrlKey && event.shiftKey && event.key === 'S') {
        event.preventDefault();
        toggleSimulation();
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();

    // Start demo automatically
    fetch('/demo/start', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log('Demo started:', data.message))
        .catch(err => console.error('Error starting demo:', err));
});
