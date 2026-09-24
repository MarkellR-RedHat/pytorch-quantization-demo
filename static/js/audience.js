let selectedModel = 'FP16';
let yourRequestCount = 0;
let ws = null;

const modelButtons = document.querySelectorAll('.model-btn');
const sendButton = document.getElementById('sendRequestBtn');
const btnSubtext = document.getElementById('btnSubtext');
const yourRequestsEl = document.getElementById('yourRequests');
const totalRequestsEl = document.getElementById('totalRequests');
const participantsEl = document.getElementById('participants');
const statusMessageEl = document.getElementById('statusMessage');

const DISPLAY_NAMES = {
    'FP16': 'FP16',
    'INT4': 'INT4',
    'SPEC_DECODE': 'Spec Decode'
};

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        showStatus('Connected! Ready to send requests.', 'success');
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = () => {
        showStatus('Connection error. Retrying...', 'error');
    };

    ws.onclose = () => {
        showStatus('Disconnected. Reconnecting...', 'warning');
        setTimeout(initWebSocket, 3000);
    };
}

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

function updateMetrics(metrics) {
    let total = 0;
    for (const model in metrics) {
        total += metrics[model].total_requests;
    }
    totalRequestsEl.textContent = total;
}

function updateState(state) {
    participantsEl.textContent = state.participant_count;
    totalRequestsEl.textContent = state.total_requests;
}

modelButtons.forEach(button => {
    button.addEventListener('click', () => {
        modelButtons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');
        selectedModel = button.dataset.model;
        btnSubtext.textContent = `${DISPLAY_NAMES[selectedModel]} selected — Tap away!`;
        showStatus(`Switched to ${DISPLAY_NAMES[selectedModel]}!`, 'success');
    });
});

let pendingRequests = 0;
const MAX_CONCURRENT = 5;

sendButton.addEventListener('click', async () => {
    if (pendingRequests >= MAX_CONCURRENT) return;

    pendingRequests++;
    yourRequestCount++;
    yourRequestsEl.textContent = yourRequestCount;

    fetch('/request', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model_type: selectedModel,
            prompt: 'Hello, how are you?'
        })
    })
    .then(response => response.json())
    .then(() => {
        showStatus(`${yourRequestCount} sent to ${DISPLAY_NAMES[selectedModel]}`, 'success');
    })
    .catch(error => {
        console.error('Error:', error);
    })
    .finally(() => {
        pendingRequests--;
    });
});

function showStatus(message, type = 'info') {
    statusMessageEl.textContent = message;
    statusMessageEl.className = `status-message ${type}`;
}

initWebSocket();
showStatus('FP16 pre-selected — Start tapping!', 'success');
