/* Quantization Arena booth page: the game and its scoreboard. */
(function () {
    'use strict';
    const $ = (s, el = document) => el.querySelector(s);
    const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

    const BIRD_INFO = {
        SPEC: { name: 'Spec Decode', color: '--v-spec', sub: 'BF16 target + 4-unit draft' },
        BF16: { name: 'BF16', color: '--v-bf16', sub: '16-bit weights' },
        FP8: { name: 'FP8', color: '--v-fp8', sub: '8-bit float, one scale per row' },
        INT4_AWQ: { name: 'INT4 AWQ', color: '--v-int4', sub: '4-bit, activation-aware scaling' },
        INT4_RTN: { name: 'INT4 RTN', color: '--v-int4', sub: '4-bit, plain rounding', hollow: true },
    };

    let config = null;
    let arena = null;
    let ws = null;
    let votes = {};
    let lastSnapshot = null;

    // stage scaling

    function fit() {
        const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
        const stage = $('#stage');
        stage.style.transform = `scale(${s})`;
        stage.style.left = `${(window.innerWidth - 1920 * s) / 2}px`;
        stage.style.top = `${(window.innerHeight - 1080 * s) / 2}px`;
        if (arena) {
            const r = $('#arenaCanvas').getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            arena.resize(Math.round(r.width * dpr), Math.round(r.height * dpr));
        }
    }
    window.addEventListener('resize', fit);

    // theme

    function setTheme(t) {
        document.documentElement.dataset.theme = t;
        try { localStorage.setItem('qs-theme', t); } catch (e) { /* private mode */ }
        if (arena) arena.readColors();
    }
    try { const t = localStorage.getItem('qs-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

    // arena

    async function startArena() {
        const res = await fetch('/static/arena/policy.json');
        const policy = await res.json();
        $('#paramCount').textContent = policy.params.target.toLocaleString('en-US');
        arena = new window.QuantArena.Arena($('#arenaCanvas'), policy, {
            onChange: snap => { lastSnapshot = snap; renderFlock(snap); renderSpec(snap); },
            onEvent: ev => { if (ev.type === 'crash') flashRow(ev.bird); },
        });
        buildFlock();
        $('#arenaLoading').hidden = true;
        fit();
        setInterval(sendArenaState, 500);
    }

    function buildFlock() {
        const ol = $('#flockRows');
        ol.innerHTML = '';
        for (const key of ['SPEC', 'BF16', 'FP8', 'INT4_AWQ', 'INT4_RTN']) {
            const info = BIRD_INFO[key];
            const li = document.createElement('li');
            li.className = 'row';
            li.dataset.key = key;
            li.style.setProperty('--c', `var(${info.color})`);
            li.innerHTML = `
                <span class="glyph${info.hollow ? ' hollow' : ''}" style="--c: var(${info.color})"></span>
                <span class="who"><b>${info.name}</b><span class="sub">${info.sub}</span></span>
                <span class="score"><b class="num">–</b><span class="num">0/0 · 0 crashes</span></span>
                <span class="meter"><i style="width:0%"></i></span>`;
            li.addEventListener('click', () => { arena.toggle(key); li.classList.toggle('is-hidden'); });
            ol.appendChild(li);
        }
    }

    function renderFlock(snap) {
        for (const li of $$('#flockRows .row')) {
            const b = snap.birds[li.dataset.key];
            if (!b) continue;
            const [c, a] = b.hard;
            const pct = a ? Math.round(100 * c / a) : null;
            $('.score b', li).textContent = pct === null ? '–' : `${pct}%`;
            const backers = votes[li.dataset.key] || 0;
            $('.score span', li).textContent = `${c}/${a} · ${b.crashes} crash${b.crashes === 1 ? '' : 'es'}`;
            $('.meter i', li).style.width = `${pct === null ? 0 : pct}%`;
            const info = BIRD_INFO[li.dataset.key];
            $('.who .sub', li).textContent = backers ? `${info.sub} · ${backers} backing` : info.sub;
        }
    }

    function flashRow(key) {
        const li = $(`#flockRows .row[data-key="${key}"]`);
        if (!li) return;
        li.classList.remove('flash');
        void li.offsetWidth;
        li.classList.add('flash');
    }

    function renderSpec(snap) {
        const s = snap.spec;
        if (!s) return;
        $('#specAccept').textContent = `${Math.round(100 * s.acceptance)}%`;
        $('#specFpp').textContent = s.frames_per_pass.toFixed(1);
        $('#specDiff').textContent = `${s.max_diff_px.toFixed(0)} px`;
        $('#specK').textContent = `draft guesses ${s.k} moves`;
        const box = $('#specStrips');
        box.innerHTML = '';
        for (const st of s.strips) {
            const col = document.createElement('div');
            col.className = 'strip';
            for (let i = 0; i <= st.k; i++) {
                const cell = document.createElement('i');
                if (i < st.accepted) cell.className = 'ok';
                else if (i === st.accepted) cell.className = 'fix';
                col.appendChild(cell);
            }
            box.appendChild(col);
        }
    }

    function sendArenaState() {
        if (!ws || ws.readyState !== 1 || !lastSnapshot) return;
        const s = lastSnapshot;
        const data = { gaps: s.gaps, birds: s.birds, spec: s.spec ? { acceptance: s.spec.acceptance, frames_per_pass: s.spec.frames_per_pass, max_diff_px: s.spec.max_diff_px } : null };
        ws.send(JSON.stringify({ type: 'arena_state', data }));
    }

    function feed(html, cls) {
        const ol = $('#feed');
        const li = document.createElement('li');
        if (cls) li.className = cls;
        li.innerHTML = `<span class="dot"></span><span>${html}</span>`;
        ol.prepend(li);
        while (ol.children.length > 3) ol.lastChild.remove();
        setTimeout(() => li.classList.add('fade'), 7000);
        setTimeout(() => li.remove(), 7700);
    }

    function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

    function hardPrompt(label, from) {
        if (!arena) return;
        arena.throwHard(label, from);
        feed(`<b>${esc(from || 'Presenter')}</b> threw <b>${esc(label)}</b>`);
    }

    // server

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws/presenter`);
        ws.onmessage = ev => {
            let msg;
            try { msg = JSON.parse(ev.data); } catch (e) { return; }
            const d = msg.data;
            switch (msg.type) {
                case 'arena_hard_prompt': hardPrompt(d.label, d.from); break;
                case 'arena_votes': votes = d.counts || d || {}; if (lastSnapshot) renderFlock(lastSnapshot); break;
                default: break;
            }
        };
        ws.onclose = () => setTimeout(connect, 1500);
    }

    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            config = await res.json();
        } catch (e) { config = null; }
    }

    // keyboard

    document.addEventListener('keydown', e => {
        if (e.target.closest('input, textarea')) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        switch (e.code) {
            case 'KeyT': setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); break;
            case 'KeyF':
                if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
                break;
            default:
                if (!arena) return;
                if (e.code === 'Space') { e.preventDefault(); arena.flapHuman(); }
                else if (e.code === 'KeyH') hardPrompt(['Multi-step math', 'Logic puzzle', 'Tricky code', 'Long context'][Math.floor(Math.random() * 4)], 'Presenter');
                else if (e.code === 'KeyP') { arena.userPaused = !arena.paused; arena.paused = arena.userPaused; }
                else if (e.code === 'KeyR') { arena.reset((Date.now() & 0xffff) + 1); arena.removeHuman(); }
                else if (e.code === 'Escape') arena.removeHuman();
        }
    });

    // boot

    fit();
    loadConfig();
    connect();
    startArena().catch(err => { $('#arenaLoading').textContent = `Could not load the arena: ${err.message}`; });
})();
