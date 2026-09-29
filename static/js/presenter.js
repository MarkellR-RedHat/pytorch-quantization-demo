let ws = null;
let demoRunning = false;
let challengeVisible = false;
let typingAnimations = [];

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/presenter`;

    ws = new WebSocket(wsUrl);
    ws.onopen = () => { fetchInitialState(); };
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'metrics_update') updateMetricsDisplay(msg.data);
        if (msg.type === 'state_update') updateStateDisplay(msg.data);
    };
    ws.onclose = () => { setTimeout(initWebSocket, 3000); };
}

async function fetchInitialState() {
    try {
        const [metricsRes, stateRes] = await Promise.all([fetch('/metrics'), fetch('/state')]);
        updateMetricsDisplay(await metricsRes.json());
        const state = await stateRes.json();
        updateStateDisplay(state);
        demoRunning = state.is_running;
        updateStartStopButton();
    } catch (e) { console.error(e); }
}

function updateMetricsDisplay(metrics) {
    for (const [modelType, m] of Object.entries(metrics)) {
        const card = document.querySelector(`.metric-card[data-model="${modelType}"]`);
        if (!card) continue;
        const q = (sel) => card.querySelector(sel);
        const heroLat = q('.hero-stat [data-metric="latency"]');
        const heroMem = q('.hero-stat [data-metric="memory"]');
        if (heroLat) heroLat.textContent = Math.round(m.avg_latency_ms);
        if (heroMem) heroMem.textContent = m.gpu_memory_gb.toFixed(1);
        const rps = q('[data-metric="rps"]');
        const p95 = q('[data-metric="p95"]');
        const tps = q('[data-metric="tps"]');
        const cost = q('[data-metric="cost"]');
        const total = q('[data-metric="total"]');
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

function updateStartStopButton() {
    const btn = document.getElementById('startStopBtn');
    if (demoRunning) {
        btn.textContent = 'Stop Demo';
        btn.classList.remove('start-btn');
        btn.classList.add('stop-btn');
    } else {
        btn.textContent = 'Start Demo';
        btn.classList.remove('stop-btn');
        btn.classList.add('start-btn');
    }
}

async function toggleDemo() {
    try {
        if (demoRunning) {
            await fetch('/demo/stop', { method: 'POST' });
            demoRunning = false;
            if (window.gameStop) window.gameStop();
        } else {
            await fetch('/demo/start', { method: 'POST' });
            demoRunning = true;
            if (window.gameStart) window.gameStart();
        }
        updateStartStopButton();
    } catch (e) { console.error(e); }
}

async function resetDemo() {
    if (!confirm('Reset all metrics?')) return;
    try {
        await fetch('/demo/reset', { method: 'POST' });
        demoRunning = false;
        if (window.gameReset) window.gameReset();
        updateStartStopButton();
    } catch (e) { console.error(e); }
}

function toggleChallengePanel() {
    const panel = document.getElementById('challengePanel');
    challengeVisible = !challengeVisible;
    panel.style.display = challengeVisible ? 'block' : 'none';
}

const TYPING_SPEEDS = { FP16: 47, INT4: 44, SPEC_DECODE: 40 };
const QUALITY_FAILS = {
    complex_reasoning: ['INT4'],
    code_generation: ['INT4'],
    summarization: ['INT4'],
};

function clearTyping() {
    typingAnimations.forEach(id => cancelAnimationFrame(id));
    typingAnimations = [];
    ['FP16', 'INT4', 'SPEC'].forEach(k => {
        const el = document.getElementById('output' + k);
        const vEl = document.getElementById('verdict' + k);
        if (el) el.innerHTML = '';
        if (vEl) { vEl.textContent = ''; vEl.className = 'challenge-verdict-bar'; }
    });
    document.querySelectorAll('.challenge-col').forEach(col => {
        col.classList.remove('pass', 'fail');
    });
}

async function fireChallenge(scenario) {
    clearTyping();

    document.querySelectorAll('.challenge-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-challenge="${scenario}"]`).classList.add('active');

    try {
        const res = await fetch(`/quality/${scenario}`);
        const data = await res.json();

        document.getElementById('challengePromptText').textContent = data.prompt;
        const failModels = QUALITY_FAILS[scenario] || [];

        const models = [
            { key: 'FP16', elId: 'outputFP16', verdictId: 'verdictFP16', colModel: 'FP16', speed: TYPING_SPEEDS.FP16, text: data.responses.FP16 },
            { key: 'INT4', elId: 'outputINT4', verdictId: 'verdictINT4', colModel: 'INT4', speed: TYPING_SPEEDS.INT4, text: data.responses.INT4 },
            { key: 'SPEC_DECODE', elId: 'outputSPEC', verdictId: 'verdictSPEC', colModel: 'SPEC_DECODE', speed: TYPING_SPEEDS.SPEC_DECODE, text: data.responses.SPEC_DECODE },
        ];

        models.forEach(m => {
            const el = document.getElementById(m.elId);
            const verdictEl = document.getElementById(m.verdictId);
            const col = document.querySelector(`.challenge-col[data-model="${m.colModel}"]`);
            const charsPerFrame = m.speed / 12;
            let charIndex = 0;

            function typeStep() {
                const burst = Math.max(1, Math.round(charsPerFrame + (Math.random() - 0.5) * 2));
                charIndex = Math.min(charIndex + burst, m.text.length);

                el.innerHTML = m.text.substring(0, charIndex) +
                    (charIndex < m.text.length ? '<span class="typing-cursor"></span>' : '');

                if (charIndex < m.text.length) {
                    const id = requestAnimationFrame(typeStep);
                    typingAnimations.push(id);
                } else {
                    const failed = failModels.includes(m.key);
                    verdictEl.textContent = failed ? 'QUALITY FAIL — Wrong Answer' : 'PASSED — Correct';
                    verdictEl.classList.add('show', failed ? 'fail-bar' : 'pass-bar');
                    col.classList.add(failed ? 'fail' : 'pass');
                }
            }

            const delay = (TYPING_SPEEDS.FP16 - m.speed) * 40;
            setTimeout(() => {
                const id = requestAnimationFrame(typeStep);
                typingAnimations.push(id);
            }, delay);
        });
    } catch (e) { console.error(e); }
}

document.getElementById('startStopBtn').addEventListener('click', toggleDemo);
document.getElementById('resetBtn').addEventListener('click', resetDemo);
document.getElementById('challengeBtn').addEventListener('click', toggleChallengePanel);

document.querySelectorAll('.challenge-btn').forEach(btn => {
    btn.addEventListener('click', () => fireChallenge(btn.dataset.challenge));
});

document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'S') { e.preventDefault(); toggleDemo(); }
    if (e.ctrlKey && e.shiftKey && e.key === 'Q') { e.preventDefault(); toggleChallengePanel(); }
});

document.addEventListener('DOMContentLoaded', () => { initWebSocket(); });
