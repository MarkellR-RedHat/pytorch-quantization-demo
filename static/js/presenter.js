(function () {
    'use strict';
    const $ = (s, el = document) => el.querySelector(s);
    const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

    const VARIANT_STYLE = {
        FP16: { color: '--v-bf16' },
        FP8: { color: '--v-fp8' },
        INT4: { color: '--v-int4' },
        SPEC_DECODE: { color: '--v-spec' },
    };
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
    let running = false;
    let votes = {};
    let lastMetrics = null;
    let lastSnapshot = null;
    let scene = 'arena';

    // ------------------------------------------------------------ stage scaling

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

    // ------------------------------------------------------------ theme

    function setTheme(t) {
        document.documentElement.dataset.theme = t;
        try { localStorage.setItem('qs-theme', t); } catch (e) { /* private mode */ }
        if (arena) arena.readColors();
    }
    try { const t = localStorage.getItem('qs-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

    // ------------------------------------------------------------ scenes

    function show(name) {
        scene = name;
        $$('.scene').forEach(s => s.classList.toggle('is-active', s.dataset.scene === name));
        $$('.tab').forEach(t => t.classList.toggle('is-active', t.dataset.scene === name));
        if (arena) arena.paused = name !== 'arena' ? true : arena.userPaused || false;
        if (name === 'numbers') renderNumbers();
        fit();
    }
    $$('.tab').forEach(t => t.addEventListener('click', () => { t.blur(); show(t.dataset.scene); }));

    // ------------------------------------------------------------ arena

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

    // ------------------------------------------------------------ numbers

    function variantKeys() {
        const keys = ['FP16', 'FP8', 'INT4', 'SPEC_DECODE'];
        const have = new Set([...(config ? config.variants.map(v => v.key) : []), ...Object.keys(lastMetrics || {})]);
        return keys.filter(k => have.has(k));
    }

    function bench(key) {
        return (config && config.benchmark && config.benchmark.variants && config.benchmark.variants[key]) || null;
    }

    function label(key) {
        const v = config && config.variants.find(x => x.key === key);
        if (v) return v.label;
        return { FP16: 'BF16', FP8: 'FP8', INT4: 'INT4 AWQ', SPEC_DECODE: 'Spec Decode' }[key] || key;
    }

    function fmtSec(ms) { return ms ? `${(ms / 1000).toFixed(2)} s` : '–'; }
    function fmt1(x) { return (x || x === 0) ? x.toFixed(1) : '–'; }

    function renderNumbers() {
        const keys = variantKeys();
        const grid = $('#variantGrid');
        grid.style.setProperty('--cols', keys.length || 3);
        $('#qualityGrid').style.setProperty('--cols', keys.length || 3);
        grid.innerHTML = '';
        const perGpu = [];
        for (const key of keys) {
            const m = lastMetrics && lastMetrics[key];
            const b = bench(key);
            const traffic = m && m.total_requests > 0;
            const live = m && m.source === 'live';
            const gpus = (b && b.gpus) || (m && m.gpus) || 1;
            const tps = b ? b.throughput_tps : null;
            const tpg = b ? (b.tokens_per_second_per_gpu || tps / gpus) : null;
            const c = (VARIANT_STYLE[key] || {}).color || '--ink-3';
            perGpu.push({ key, tpg, gpus, c });
            const card = document.createElement('div');
            card.className = 'card vcard';
            card.style.setProperty('--c', `var(${c})`);
            const cost = traffic && m.cost_per_request != null && config
                ? `<li><span>Cost per request at $${config.gpu_hourly_usd}/GPU-hr</span><b class="num">$${m.cost_per_request.toFixed(4)}</b></li>` : '';
            const row = (label, value) => `<li class="${traffic ? '' : 'muted'}"><span>${label}</span><b class="num">${traffic ? value : '–'}</b></li>`;
            card.innerHTML = `
                <div class="vcard-head">
                    <h3><span class="glyph" style="--c: var(${c})"></span>${esc(label(key))}</h3>
                    <span class="basis basis-bench">Benchmark · 1 stream</span>
                </div>
                <div class="gpus">${'<i></i>'.repeat(gpus)}<span>${gpus} × H200</span></div>
                <div class="hero">
                    <div><b class="num">${fmt1(tps)}</b><span>tokens/s per stream</span></div>
                    <div><b class="num">${fmt1(tpg)}</b><span>tokens/s per GPU</span></div>
                </div>
                <ul class="stats">
                    <li><span>Mean · p95 time per request</span><b class="num">${b ? `${fmtSec(b.avg_latency_ms)} · ${fmtSec(b.p95_latency_ms)}` : '–'}</b></li>
                </ul>
                <div class="now-head">
                    <h4>Right now</h4>
                    <span class="basis ${live ? 'basis-live' : 'basis-sim'}">${live ? 'Live' : 'Simulated'}</span>
                </div>
                <ul class="stats">
                    ${row('Requests in flight', m ? m.in_flight : 0)}
                    ${row('Requests served', m ? `${m.total_requests.toLocaleString('en-US')}${m.errors ? ` · ${m.errors} errors` : ''}` : 0)}
                    ${row('Median · p95 time per request', m ? `${fmtSec(m.p50_latency_ms)} · ${fmtSec(m.p95_latency_ms)}` : '')}
                    ${cost}
                </ul>`;
            grid.appendChild(card);
        }
        const max = Math.max(...perGpu.map(p => p.tpg || 0), 1);
        const bars = $('#perGpuBars');
        bars.innerHTML = '';
        for (const p of perGpu) {
            const el = document.createElement('div');
            el.className = 'bar';
            el.style.setProperty('--c', `var(${p.c})`);
            el.innerHTML = `
                <span class="lab"><span class="glyph" style="--c: var(${p.c})"></span>${esc(label(p.key))}</span>
                <span class="track"><span class="fill" style="width:${(100 * (p.tpg || 0) / max).toFixed(1)}%"></span></span>
                <span class="val num">${fmt1(p.tpg)}<small>on ${p.gpus} GPU${p.gpus > 1 ? 's' : ''}</small></span>`;
            bars.appendChild(el);
        }
        const bf = perGpu.find(p => p.key === 'FP16'), q = perGpu.find(p => p.key === 'INT4');
        if (bf && q && bf.tpg && q.tpg) {
            const ratio = q.tpg / bf.tpg, speed = bench('INT4').throughput_tps / bench('FP16').throughput_tps;
            const t = document.createElement('p');
            t.className = 'takeaway';
            t.innerHTML = `${esc(label('INT4'))} gets <b>${ratio.toFixed(2)}×</b> the tokens per GPU of ${esc(label('FP16'))}, because it runs at ${Math.round(100 * speed)}% of the speed on half the GPUs.`;
            bars.appendChild(t);
        }
        renderLoad();
    }

    // Line chart of output tokens/s per GPU across the vllm bench serve concurrency sweep.
    function renderLoad() {
        const body = $('#loadBody');
        const load = (config && config.benchmark && config.benchmark.load) || {};
        const series = Object.entries(load).filter(([, pts]) => pts.length > 1);
        if (!series.length) {
            body.innerHTML = `<div class="pending"><span>Not measured yet. This fills in from the <b>vllm bench serve</b> sweep at 1, 8, 32 and 64 concurrent requests, which is where cost per token is actually decided.</span></div>`;
            return;
        }
        const W = 820, H = 196, L = 56, R = 150, T = 12, B = 30;
        const xs = [...new Set(series.flatMap(([, pts]) => pts.map(p => p.concurrency)))].sort((a, b) => a - b);
        const ymax = Math.max(...series.flatMap(([, pts]) => pts.map(p => p.output_tokens_per_second_per_gpu))) * 1.1;
        const lx = v => L + (Math.log2(v) - Math.log2(xs[0])) / Math.max(1e-9, Math.log2(xs[xs.length - 1]) - Math.log2(xs[0])) * (W - L - R);
        const ly = v => T + (1 - v / ymax) * (H - T - B);
        let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Output tokens per second per GPU by concurrency">`;
        for (let i = 0; i <= 4; i++) {
            const v = ymax * i / 4, y = ly(v);
            svg += `<line class="gridline" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="axis" x="${L - 10}" y="${y + 4}" text-anchor="end">${Math.round(v)}</text>`;
        }
        for (const x of xs) svg += `<text class="axis" x="${lx(x)}" y="${H - 10}" text-anchor="middle">${x}</text>`;
        const ends = [];
        for (const [key, pts] of series) {
            const col = `var(${(VARIANT_STYLE[key] || {}).color || '--ink-3'})`;
            svg += `<polyline class="series" style="stroke:${col}" points="${pts.map(p => `${lx(p.concurrency)},${ly(p.output_tokens_per_second_per_gpu)}`).join(' ')}"/>`;
            for (const p of pts) svg += `<circle class="dot" style="fill:${col}" cx="${lx(p.concurrency)}" cy="${ly(p.output_tokens_per_second_per_gpu)}" r="5"><title>${esc(label(key))}: ${p.output_tokens_per_second_per_gpu} tok/s per GPU at ${p.concurrency} concurrent</title></circle>`;
            const last = pts[pts.length - 1];
            ends.push({ y: ly(last.output_tokens_per_second_per_gpu), key });
        }
        ends.sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 20) ends[i].y = ends[i - 1].y + 20;
        for (const e of ends) svg += `<text class="end" x="${W - R + 12}" y="${e.y + 5}">${esc(label(e.key))}</text>`;
        svg += '</svg>';
        body.innerHTML = svg;
        $('#loadHint').textContent = 'output tokens/s per GPU · concurrent requests →';
    }

    // ------------------------------------------------------------ quality

    const CHECK = '<svg viewBox="0 0 24 24"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>';
    const CROSS = '<svg viewBox="0 0 24 24"><path d="M19 6.4 17.6 5 12 10.6 6.4 5 5 6.4 10.6 12 5 17.6 6.4 19 12 13.4 17.6 19 19 17.6 13.4 12z"/></svg>';
    let typing = [];

    async function fireQuality(scenario) {
        typing.forEach(cancelAnimationFrame);
        typing = [];
        $$('#qualityPick button').forEach(b => b.classList.toggle('is-active', b.dataset.scenario === scenario));
        const res = await fetch(`/quality/${scenario}`);
        const data = await res.json();
        const captured = data.source === 'captured';
        $('#qualitySource').innerHTML = captured
            ? `Captured from the real deployments at temperature ${data.temperature ?? 0}. Nothing on this screen is edited.`
            : 'Illustrative outputs written to show the failure mode. Real captures from the H200 deployments replace these automatically once they are recorded.';
        $('#qualityPrompt').textContent = data.prompt;
        const grid = $('#qualityGrid');
        grid.innerHTML = '';
        const keys = Object.keys(data.responses);
        grid.style.setProperty('--cols', keys.length);
        for (const key of ['FP16', 'FP8', 'INT4', 'SPEC_DECODE'].filter(k => keys.includes(k))) {
            const r = typeof data.responses[key] === 'string' ? { text: data.responses[key] } : data.responses[key];
            const c = (VARIANT_STYLE[key] || {}).color || '--ink-3';
            const col = document.createElement('div');
            col.className = 'card qcol';
            col.innerHTML = `
                <div class="card-head"><h2><span class="glyph" style="--c: var(${c})"></span>${esc(label(key))}</h2>
                <span class="src-badge ${captured ? 'captured' : 'illustrative'}">${captured ? 'captured' : 'illustrative'}</span></div>
                <pre class="qout"></pre>
                <div class="verdict"></div>`;
            grid.appendChild(col);
            typeOut(col, r, key);
        }
    }

    function typeOut(col, r, key) {
        const out = $('.qout', col), verdict = $('.verdict', col);
        const b = bench(key);
        // Relative typing speed follows the measured single-stream tokens/s, about 4 chars per token.
        const tps = (b && b.throughput_tps) || 40;
        const cps = tps * 4 * 1.6;
        // Model output is markdown; show it as plain text without fences or bold markers.
        const text = (r.text || '').replace(/^```[a-z]*\n?/gim, '').replace(/\*\*(.+?)\*\*/g, '$1').trim();
        const t0 = performance.now();
        function step(t) {
            const n = Math.min(text.length, Math.floor((t - t0) / 1000 * cps));
            out.textContent = text.slice(0, n);
            if (n < text.length) {
                const cur = document.createElement('span');
                cur.className = 'cursor';
                out.appendChild(cur);
                typing.push(requestAnimationFrame(step));
            } else if (r.verdict) {
                verdict.className = `verdict ${r.verdict}`;
                verdict.innerHTML = r.verdict === 'pass' ? `${CHECK}Correct answer` : `${CROSS}Wrong answer`;
            }
        }
        typing.push(requestAnimationFrame(step));
    }
    $$('#qualityPick button').forEach(b => b.addEventListener('click', () => fireQuality(b.dataset.scenario)));

    // ------------------------------------------------------------ server

    async function control(path) {
        const res = await fetch(path, { method: 'POST', credentials: 'same-origin' });
        if (res.status === 401 || res.status === 403) {
            feed('Presenter key needed. Open <b>/presenter?key=…</b> on this laptop.', 'warn');
            return null;
        }
        return res.ok ? res.json().catch(() => ({})) : null;
    }

    function setRunning(r) {
        running = r;
        const btn = $('#startStopBtn');
        btn.textContent = r ? 'Stop traffic' : 'Start traffic';
        btn.classList.toggle('is-running', r);
    }

    async function toggleTraffic() {
        $('#startStopBtn').blur();
        const r = await control(running ? '/demo/stop' : '/demo/start');
        if (r) setRunning(!running);
    }
    $('#startStopBtn').addEventListener('click', toggleTraffic);
    $('#resetBtn').addEventListener('click', async () => {
        $('#resetBtn').blur();
        if (!window.confirm('Reset traffic metrics and votes?')) return;
        if (await control('/demo/reset')) { setRunning(false); if (arena) arena.reset(Date.now() & 0xffff); }
    });

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws/presenter`);
        ws.onmessage = ev => {
            let msg;
            try { msg = JSON.parse(ev.data); } catch (e) { return; }
            const d = msg.data;
            switch (msg.type) {
                case 'metrics_update': lastMetrics = d; if (scene === 'numbers') renderNumbers(); break;
                case 'state_update':
                    $('#participants').textContent = d.participant_count ?? 0;
                    if (typeof d.is_running === 'boolean') setRunning(d.is_running);
                    break;
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
            if (!res.ok) throw new Error(res.status);
            config = await res.json();
        } catch (e) {
            config = null;
        }
        const badge = $('#modeBadge');
        if (config) {
            const live = config.mode === 'live';
            badge.className = `mode ${live ? 'live' : 'sim'}`;
            $('span', badge).textContent = live ? 'Live models' : 'Simulated · benchmark data';
            const url = config.public_url || location.origin + '/';
            $('#joinUrl').textContent = url.replace(/^https?:\/\//, '').replace(/\/$/, '');
            const bm = config.benchmark || {};
            if (bm.model) $('#subtitle').textContent = `${bm.model.split('/').pop().replace(/-/g, ' ')} on vLLM · ${bm.gpu || 'NVIDIA H200'}`;
            const when = bm.date ? new Date(bm.date + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '';
            $('#benchNote').textContent = `Measured ${when} on ${bm.gpu || 'NVIDIA H200'} with vLLM ${bm.vllm_version || ''}, one request at a time (20 per variant, up to 256 output tokens), so these are single-stream speeds and not server capacity under load.`;
        } else {
            badge.className = 'mode';
            $('span', badge).textContent = 'Offline';
            $('#joinUrl').textContent = location.host;
        }
        if (scene === 'numbers') renderNumbers();
    }

    // ------------------------------------------------------------ keyboard

    document.addEventListener('keydown', e => {
        if (e.target.closest('input, textarea')) return;
        if (e.ctrlKey && e.shiftKey && e.code === 'KeyS') { e.preventDefault(); toggleTraffic(); return; }
        if (e.ctrlKey && e.shiftKey && e.code === 'KeyQ') { e.preventDefault(); show('quality'); return; }
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        switch (e.code) {
            case 'Digit1': show('arena'); break;
            case 'Digit2': show('numbers'); break;
            case 'Digit3': show('quality'); break;
            case 'KeyT': setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); break;
            case 'KeyF':
                if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
                break;
            default:
                if (scene !== 'arena' || !arena) return;
                if (e.code === 'Space') { e.preventDefault(); arena.flapHuman(); }
                else if (e.code === 'KeyH') hardPrompt(['Multi-step math', 'Logic puzzle', 'Tricky code', 'Long context'][Math.floor(Math.random() * 4)], 'Presenter');
                else if (e.code === 'KeyP') { arena.userPaused = !arena.paused; arena.paused = arena.userPaused; }
                else if (e.code === 'KeyR') { arena.reset((Date.now() & 0xffff) + 1); arena.removeHuman(); }
                else if (e.code === 'Escape') arena.removeHuman();
        }
    });

    // ------------------------------------------------------------ boot

    fit();
    loadConfig();
    connect();
    startArena().catch(err => { $('#arenaLoading').textContent = `Could not load the arena: ${err.message}`; });
})();
