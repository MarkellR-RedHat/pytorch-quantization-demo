let ws = null;
let simulationMode = false;
let qualityVisible = false;

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

function updateMetricsDisplay(metrics) {
    for (const [modelType, m] of Object.entries(metrics)) {
        const card = document.querySelector(`.metric-card[data-model="${modelType}"]`);
        if (!card) continue;

        const heroLatency = card.querySelector('.hero-stat [data-metric="latency"]');
        const heroMemory = card.querySelector('.hero-stat [data-metric="memory"]');
        if (heroLatency) heroLatency.textContent = Math.round(m.avg_latency_ms);
        if (heroMemory) heroMemory.textContent = m.gpu_memory_gb.toFixed(1);

        const rps = card.querySelector('[data-metric="rps"]');
        const p95 = card.querySelector('[data-metric="p95"]');
        const tps = card.querySelector('[data-metric="tps"]');
        const cost = card.querySelector('[data-metric="cost"]');
        const total = card.querySelector('[data-metric="total"]');

        if (rps) rps.textContent = `${m.requests_per_second.toFixed(1)} req/s`;
        if (p95) p95.textContent = `${Math.round(m.p95_latency_ms)} ms`;
        if (tps) tps.textContent = m.tokens_per_second.toFixed(1);
        if (cost) cost.textContent = `$${m.cost_per_request.toFixed(4)}`;
        if (total) total.textContent = m.total_requests;
    }
}

function updateStateDisplay(state) {
    document.getElementById('totalParticipants').textContent = state.participant_count;
    document.getElementById('totalDemoRequests').textContent = state.total_requests;
}

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

async function toggleSimulation() {
    try {
        await fetch('/simulation/toggle', { method: 'POST' });
    } catch (error) {
        console.error('Error toggling simulation:', error);
    }
}

async function resetDemo() {
    if (!confirm('Reset all metrics?')) return;
    try {
        await fetch('/demo/reset', { method: 'POST' });
    } catch (error) {
        console.error('Error resetting demo:', error);
    }
}

async function loadQualityComparison(scenario) {
    try {
        const res = await fetch(`/quality/${scenario}`);
        const data = await res.json();

        document.getElementById('qualityPrompt').textContent = data.prompt;
        document.getElementById('qualityFP16').textContent = data.responses.FP16;
        document.getElementById('qualityINT4').textContent = data.responses.INT4;
        document.getElementById('qualitySPEC').textContent = data.responses.SPEC_DECODE;
    } catch (error) {
        console.error('Error loading quality comparison:', error);
    }
}

function toggleQualityPanel() {
    const panel = document.getElementById('qualityPanel');
    qualityVisible = !qualityVisible;
    panel.style.display = qualityVisible ? 'block' : 'none';
    if (qualityVisible) {
        loadQualityComparison('complex_reasoning');
    }
}

document.getElementById('toggleSimBtn').addEventListener('click', toggleSimulation);
document.getElementById('resetBtn').addEventListener('click', resetDemo);
document.getElementById('qualityBtn').addEventListener('click', toggleQualityPanel);

document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadQualityComparison(btn.dataset.scenario);
    });
});

document.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.shiftKey && event.key === 'S') {
        event.preventDefault();
        toggleSimulation();
    }
    if (event.ctrlKey && event.shiftKey && event.key === 'Q') {
        event.preventDefault();
        toggleQualityPanel();
    }
});

document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    fetch('/demo/start', { method: 'POST' })
        .then(res => res.json())
        .then(data => console.log('Demo started:', data.message))
        .catch(err => console.error('Error starting demo:', err));
});
