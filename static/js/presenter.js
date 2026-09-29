(function () {
    'use strict';
    const $ = (s, el = document) => el.querySelector(s);
    const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

    // What each setup buys you. Wording matches the slides.
    const SETUPS = {
        FP16: {
            color: '--v-bf16', role: 'The baseline',
            gets: 'Full quality, the answer every other setup is measured against',
            best: 'Questions where a wrong answer is expensive',
            trade: 'Two GPUs for every replica, so every new replica doubles the bill',
            watch: 'Two GPUs per replica, so scaling out gets expensive fast',
            route: 'A wrong answer is expensive',
            acc: '100%', accNote: 'the baseline',
        },
        INT4: {
            color: '--v-int4', role: 'Half the GPUs',
            gets: 'Half the GPUs, at about the same speed per request',
            best: 'High-volume chat and easy questions',
            trade: 'Small accuracy loss that shows up on hard reasoning, so test it on your own prompts',
            route: 'Easy questions',
            acc: '≈99%', accNote: 'Red Hat\'s INT4 build, 97% on harder tests*',
            watch: 'Slips first on hard reasoning, so test it on your own prompts',
        },
        SPEC_DECODE: {
            color: '--v-spec', role: 'Same GPUs, built for lower latency',
            gets: 'Built for faster answers on the same GPUs, with BF16 quality',
            best: 'Latency-sensitive, low-traffic work',
            trade: 'A draft model to host next to the big one, and the gain shrinks as traffic grows',
            note: 'This run had CUDA graphs turned off (enforce_eager), which slows speculative decoding the most. The rerun without it is next.',
            route: 'Latency-sensitive, low traffic', pending: 'after the rerun',
            acc: '100%', accNote: 'the 70B checks every token',
            watch: 'Came out slower in this run, with CUDA graphs off, so it\'s being rerun before we route to it',
        },
        FP8: {
            color: '--v-fp8', role: 'One GPU, near-lossless',
            gets: '8-bit weights and activations on a single GPU',
            best: 'High-throughput serving on Hopper GPUs',
            trade: 'Needs GPUs with FP8 support',
        },
    };
    const ORDER = ['FP16', 'INT4', 'SPEC_DECODE', 'FP8'];
    const PRESET_TEXT = {
        reasoning: 'A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.',
        code: 'Write a Python function that returns the second largest number in a list. Handle edge cases.',
        summary: 'Summarize the key trade-offs of model quantization for production LLM deployments in 3 bullet points.',
    };

    let config = null;
    let running = [];
    let finished = 0;

    // ------------------------------------------------------------ stage, theme, scenes

    function fit() {
        const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
        const stage = $('#stage');
        stage.style.transform = `scale(${s})`;
        stage.style.left = `${(window.innerWidth - 1920 * s) / 2}px`;
        stage.style.top = `${(window.innerHeight - 1080 * s) / 2}px`;
    }
    window.addEventListener('resize', fit);

    function setTheme(t) {
        document.documentElement.dataset.theme = t;
        try { localStorage.setItem('qs-theme', t); } catch (e) { /* private mode */ }
    }
    try { const t = localStorage.getItem('qs-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

    let order = ['ask', 'load', 'numbers'];

    function setOrder() {
        order = loadData().keys.length ? ['ask', 'load', 'numbers'] : ['ask', 'numbers'];
        $$('.tab').forEach(t => {
            const i = order.indexOf(t.dataset.scene);
            t.hidden = i < 0;
            if (i >= 0) $('kbd', t).textContent = String(i + 1);
        });
        $('#footKeys').innerHTML = order.map((_, i) => `<kbd>${i + 1}</kbd>`).join('');
    }

    function show(name) {
        $$('.scene').forEach(s => s.classList.toggle('is-active', s.dataset.scene === name));
        $$('.tab').forEach(t => t.classList.toggle('is-active', t.dataset.scene === name));
    }
    $$('.tab').forEach(t => t.addEventListener('click', () => { t.blur(); show(t.dataset.scene); }));

    // ------------------------------------------------------------ helpers

    function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
    function variants() {
        const have = new Set(config ? config.variants.map(v => v.key) : ['FP16', 'INT4', 'SPEC_DECODE']);
        return ORDER.filter(k => have.has(k));
    }
    function label(key) {
        const v = config && config.variants.find(x => x.key === key);
        return v ? v.label : ({ FP16: 'BF16', FP8: 'FP8', INT4: 'INT4 AWQ', SPEC_DECODE: 'Spec Decode' }[key] || key);
    }
    function bench(key) { return (config && config.benchmark && config.benchmark.variants && config.benchmark.variants[key]) || null; }
    function gpus(key) {
        const b = bench(key), v = config && config.variants.find(x => x.key === key);
        return (b && b.gpus) || (v && v.gpus) || (key === 'INT4' || key === 'FP8' ? 1 : 2);
    }
    function color(key) { return (SETUPS[key] || {}).color || '--ink-3'; }
    function chips(key) { const n = gpus(key); return `<span class="gpu-chips">${'<i></i>'.repeat(n)}${n} × H200</span>`; }
    function glyph(key) { return `<span class="glyph" style="--c: var(${color(key)})"></span>`; }
    function markdownToText(t) { return t.replace(/^```[a-z]*\n?/gim, '').replace(/\*\*(.+?)\*\*/g, '$1'); }
    const one = x => (Math.round(x * 10 + 1e-6) / 10).toFixed(1);
    const secs = ms => `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;

    // ------------------------------------------------------------ example router

    // A deliberately simple, visible rule that shows where a router fits. It isn't a trained classifier.
    function route(q) {
        const t = q.toLowerCase();
        if (/\b(legal|contract|diagnos\w*|analy[sz]e|compliance|medical|financial)\b/.test(t)) {
            return { key: 'FP16', why: 'a wrong answer here is expensive' };
        }
        if (/\b(debug|code|function|python|sql|regex|script)\b/.test(t)) {
            return { key: 'FP16', why: 'it\'s code, where one wrong token breaks the answer' };
        }
        if (/\b(why|prove|step by step|how many|calculate|compare|reason\w*|riddle|puzzle)\b/.test(t)) {
            return { key: 'FP16', why: 'it\'s a multi-step reasoning question, and that\'s where 4-bit models slip first' };
        }
        if (/\b(write|draft|explain|describe|story|essay|report)\b/.test(t) && t.length > 60) {
            return { key: 'SPEC_DECODE', why: 'it\'s a long answer and someone is waiting on it' };
        }
        return { key: 'INT4', why: 'it\'s an everyday question, so the cheapest tokens win' };
    }

    // ------------------------------------------------------------ ask

    function buildAsk() {
        const keys = variants();
        const grid = $('#askGrid');
        grid.style.setProperty('--cols', keys.length);
        grid.innerHTML = '';
        for (const key of keys) {
            const col = document.createElement('div');
            col.className = 'card acol';
            col.dataset.key = key;
            col.style.setProperty('--c', `var(${color(key)})`);
            col.innerHTML = `
                <div class="acol-head">
                    <div class="row1"><h3>${glyph(key)}${esc(label(key))}</h3>${chips(key)}</div>
                    <p class="role">${esc((SETUPS[key] || {}).role || '')}</p>
                </div>
                <pre class="answer idle">Waiting for a question.</pre>
                <div class="astats">
                    <div><b class="num s-ttft">–</b><span>first token</span></div>
                    <div><b class="num s-tps">–</b><span>tokens/s</span></div>
                    <div><b class="num s-len">–</b><span>tokens in answer</span></div>
                    <div><b class="num s-total">–</b><span>total</span></div>
                    <span class="place"></span>
                </div>`;
            grid.appendChild(col);
        }
    }

    function ask(prompt, preset) {
        running.forEach(c => c.abort());
        running = [];
        finished = 0;
        const r = route(prompt);
        const router = $('#router');
        router.hidden = false;
        router.innerHTML = `An example routing rule sends this to ${glyph(r.key)}<b>${esc(label(r.key))}</b>, because ${esc(r.why)}`;
        for (const col of $$('#askGrid .acol')) {
            col.classList.toggle('is-routed', col.dataset.key === r.key);
            streamInto(col, prompt, preset);
        }
    }

    async function streamInto(col, prompt, preset) {
        const key = col.dataset.key, out = $('.answer', col), place = $('.place', col);
        out.className = 'answer';
        out.textContent = '';
        place.textContent = '';
        place.className = 'place';
        $('.s-ttft', col).textContent = '…';
        $('.s-tps', col).textContent = '…';
        $('.s-len', col).textContent = '…';
        const ctrl = new AbortController();
        running.push(ctrl);
        const t0 = performance.now();
        const tick = setInterval(() => { $('.s-total', col).textContent = secs(performance.now() - t0); }, 100);
        let raw = '';
        try {
            const res = await fetch(`/ask/${key}`, {
                method: 'POST', credentials: 'same-origin', signal: ctrl.signal,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, preset: preset || null }),
            });
            if (res.status === 401 || res.status === 403) throw new Error('Open /presenter?key=… on this laptop first.');
            if (!res.ok || !res.body) throw new Error('This model didn\'t answer.');
            const reader = res.body.getReader(), dec = new TextDecoder();
            let buf = '';
            for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += dec.decode(value, { stream: true });
                let nl;
                while ((nl = buf.indexOf('\n')) >= 0) {
                    const line = buf.slice(0, nl);
                    buf = buf.slice(nl + 1);
                    if (!line.trim()) continue;
                    const ev = JSON.parse(line);
                    if (ev.t === 'delta') {
                        raw += ev.text;
                        out.textContent = markdownToText(raw);
                        out.scrollTop = out.scrollHeight;
                    } else if (ev.t === 'done') {
                        clearInterval(tick);
                        // The Sep 29 benchmark measured whole requests, not time to first token, so replay can't show it.
                        const ttft = $('.s-ttft', col);
                        ttft.classList.toggle('na', ev.ttft_ms == null);
                        ttft.textContent = ev.ttft_ms != null ? secs(ev.ttft_ms) : 'live only';
                        $('.s-tps', col).textContent = ev.tokens_per_second != null ? ev.tokens_per_second.toFixed(0) : '–';
                        $('.s-total', col).textContent = secs(ev.total_ms);
                        const len = $('.s-len', col);
                        len.classList.toggle('na', ev.completion_tokens == null);
                        len.textContent = ev.completion_tokens != null ? ev.completion_tokens : 'live only';
                        finished += 1;
                        place.textContent = ['1st', '2nd', '3rd', '4th'][finished - 1] || '';
                        if (finished === 1) place.classList.add('first');
                    } else if (ev.t === 'error') {
                        throw new Error('This model didn\'t answer.');
                    }
                }
            }
        } catch (e) {
            clearInterval(tick);
            if (e.name === 'AbortError') return;
            out.className = 'answer err';
            out.textContent = e.message;
            $('.s-ttft', col).textContent = '–';
            $('.s-tps', col).textContent = '–';
            $('.s-len', col).textContent = '–';
            $('.s-total', col).textContent = '–';
        }
    }

    $('#askForm').addEventListener('submit', e => {
        e.preventDefault();
        const q = $('#askInput').value.trim();
        if (q) ask(q, null);
    });
    $$('#presets button').forEach(b => b.addEventListener('click', () => {
        b.blur();
        $('#askInput').value = PRESET_TEXT[b.dataset.preset];
        ask(PRESET_TEXT[b.dataset.preset], b.dataset.preset);
    }));

    // ------------------------------------------------------------ numbers (the money slide)

    function buildNumbers() {
        const keys = variants();
        const grid = $('#moneyGrid');
        grid.style.setProperty('--cols', keys.length);
        grid.innerHTML = '';
        for (const key of keys) {
            const s = SETUPS[key] || {}, b = bench(key), n = gpus(key);
            const tps = b ? b.throughput_tps : null;
            const tpg = b ? (b.tokens_per_second_per_gpu || tps / n) : null;
            const card = document.createElement('div');
            card.className = 'card mcard';
            card.style.setProperty('--c', `var(${color(key)})`);
            card.innerHTML = `
                <div class="mhead"><h3>${glyph(key)}${esc(label(key))}</h3>${chips(key)}</div>
                <p class="gets">${esc(s.gets || '')}</p>
                <div class="nums three">
                    <div><b class="num">${tps != null ? one(tps) : '–'}</b><span>tokens/s, one request</span></div>
                    <div><b class="num">${tpg != null ? one(tpg) : '–'}</b><span>tokens/s per GPU</span></div>
                    <div><b class="num">${esc(s.acc || '–')}</b><span>${esc(s.accNote || 'accuracy')}</span></div>
                </div>
                <dl><div><dt>Best for</dt><dd>${esc(s.best || '')}</dd></div>${s.watch ? `<div><dt>Watch out for</dt><dd>${esc(s.watch)}</dd></div>` : ''}</dl>`;
            grid.appendChild(card);
        }
        const bf = bench('FP16'), q = bench('INT4');
        if (bf && q && bf.throughput_tps && q.throughput_tps) {
            const perGpu = (q.throughput_tps / gpus('INT4')) / (bf.throughput_tps / gpus('FP16'));
            const speed = q.throughput_tps / bf.throughput_tps;
            $('#takeaway').innerHTML = `One request at a time, ${esc(label('INT4'))} gets <b>${one(perGpu)}× the tokens per GPU</b> of ${esc(label('FP16'))}, because it runs at ${Math.round(100 * speed)}% of the speed on half the GPUs.`
                + (loadData().keys.length ? '' : ' <span class="pending-note">Under heavy batching, Red Hat\'s study found 8-bit pulls ahead of 4-bit, and the load test shows where these three land.</span>');
        }
        const strip = $('#routerStrip');
        strip.className = 'card router-strip';
        strip.innerHTML = `<h3>Big, mixed traffic? Route it</h3>` + ['FP16', 'INT4', 'SPEC_DECODE'].filter(k => keys.includes(k)).map(k =>
            `<div class="route">${esc(SETUPS[k].route)} <span class="arrow">→</span> ${glyph(k)}<b>${esc(label(k))}</b>${SETUPS[k].pending ? `<em class="pend">${esc(SETUPS[k].pending)}</em>` : ''}</div>`).join('');
        const bm = (config && config.benchmark) || {};
        const when = bm.date ? new Date(bm.date + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '';
        $('#numbersFoot').textContent = '* Red Hat\'s published INT4 (GPTQ) build of Llama 3.1 70B keeps 99.4% of BF16 on the OpenLLM v1 benchmarks and 97.4% on the harder, reasoning-heavy v2 set. The AWQ build measured here still needs its own eval. Spec decode ran with CUDA graphs off (enforce_eager), which slows it the most, so it gets rerun.';
        $('#routerNote').textContent = 'A router pays off once your traffic is big and mixed enough to run more than one pool. With small traffic, pick the one setup that fits most of your questions.';
        $('#benchNote').textContent = `Llama 3.1 70B Instruct on NVIDIA H200 with vLLM ${bm.vllm_version || ''}, measured ${when}: 20 requests per setup, one at a time, up to 256 output tokens, temperature 0.7.`
            + (keys.includes('FP8') ? '' : ' FP8 fits on one H200 and is near-lossless in Red Hat\'s own tests, so it\'s the next setup worth measuring.');
    }

    // ------------------------------------------------------------ under load (replay of the load test)

    let loadTimer = null;

    function loadData() {
        const load = (config && config.benchmark && config.benchmark.load) || {};
        const keys = variants().filter(k => (load[k] || []).length > 1);
        const levels = [...new Set(keys.flatMap(k => load[k].map(p => p.concurrency)))].sort((a, b) => a - b);
        return { load, keys, levels };
    }

    function pointAt(pts, c) { return pts.find(p => p.concurrency === c) || null; }

    function buildLoad() {
        const { load, keys, levels } = loadData();
        const stage = $('#loadStage');
        stage.innerHTML = '';
        $('#loadResult').innerHTML = '';
        const play = $('#loadPlay');
        if (!keys.length) {
            play.hidden = true;
            $('#loadSetup').textContent = 'A replay of the load test, one setup at a time, as more and more questions arrive at once.';
            stage.innerHTML = '<div class="load-empty"><p>Coming soon: how each setup holds up when 1, 8, 32, then 64 questions arrive at once.</p></div>';
            return;
        }
        play.hidden = false;
        const any = load[keys[0]].find(p => p.avg_input_tokens && p.avg_output_tokens);
        const sizes = any ? ` Synthetic prompts of about ${any.avg_input_tokens} tokens, each asking for ${any.avg_output_tokens}.` : '';
        $('#loadSetup').textContent = `A replay of the vllm bench serve load test on the H200s: ${levels.join(', ').replace(/, (\d+)$/, ' and $1')} requests in flight at once, sent to each setup.${sizes}`;
        stage.style.setProperty('--cols', keys.length);
        const maxPerGpu = Math.max(...keys.flatMap(k => load[k].map(p => p.output_tokens_per_second_per_gpu)));
        for (const key of keys) {
            const card = document.createElement('div');
            card.className = 'card lcard';
            card.dataset.key = key;
            card.style.setProperty('--c', `var(${color(key)})`);
            card.innerHTML = `
                <div class="mhead"><h3>${glyph(key)}${esc(label(key))}</h3>${chips(key)}</div>
                <div class="big"><b class="num l-pergpu">–</b><span>tokens/s per GPU</span></div>
                <svg class="spark"></svg>
                <div class="lstats">
                    <div><b class="num l-total">–</b><span>tokens/s, whole setup</span></div>
                    <div><b class="num l-lat">–</b><span class="l-lat-label">median time per answer</span></div>
                </div>`;
            card.dataset.max = maxPerGpu;
            stage.appendChild(card);
        }
        showLevel(0, false);
    }

    function tween(el, to, fmt, ms) {
        const from = parseFloat(el.dataset.v || '0');
        const t0 = performance.now();
        el.dataset.v = to;
        function step(t) {
            const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3);
            el.textContent = fmt(from + (to - from) * e);
            if (k < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    function showLevel(i, animate) {
        const { load, keys, levels } = loadData();
        const ms = animate ? 900 : 1;
        for (const card of $$('#loadStage .lcard')) {
            const pts = load[card.dataset.key];
            const p = pointAt(pts, levels[i]);
            if (p) {
                tween($('.l-pergpu', card), p.output_tokens_per_second_per_gpu, v => v.toFixed(0), ms);
                tween($('.l-total', card), p.output_tokens_per_second, v => v.toFixed(0), ms);
                if (p.latency_ms) tween($('.l-lat', card), p.latency_ms / 1000, v => `${v.toFixed(1)} s`, ms);
                $('.l-lat-label', card).textContent = `${p.latency_kind === 'mean' ? 'mean' : 'median'} time per answer`;
            }
            // spark: tokens/s per GPU against concurrency, revealed up to this level
            const el = $('.spark', card);
            const W = Math.max(200, el.clientWidth), H = Math.max(100, el.clientHeight);
            el.setAttribute('viewBox', `0 0 ${W} ${H}`);
            const max = parseFloat(card.dataset.max) * 1.1, L = 12, B = 28;
            const x = j => L + j * (W - 2 * L) / Math.max(1, levels.length - 1);
            const y = v => 6 + (1 - v / max) * (H - B - 6);
            const shown = levels.slice(0, i + 1).map((c, j) => ({ j, p: pointAt(pts, c) })).filter(o => o.p);
            let svg = `<line x1="${L}" x2="${W - L}" y1="${H - B}" y2="${H - B}"/>`;
            svg += levels.map((c, j) => `<text x="${x(j)}" y="${H - 5}" text-anchor="${j === 0 ? 'start' : j === levels.length - 1 ? 'end' : 'middle'}">${c}</text>`).join('');
            if (shown.length > 1) svg += `<polyline points="${shown.map(o => `${x(o.j)},${y(o.p.output_tokens_per_second_per_gpu)}`).join(' ')}"/>`;
            svg += shown.map(o => `<circle cx="${x(o.j)}" cy="${y(o.p.output_tokens_per_second_per_gpu)}" r="7"/>`).join('');
            el.innerHTML = svg;
        }
        const play = $('#loadPlay');
        play.textContent = animate && i < levels.length - 1 ? `${levels[i]} at a time…` : 'Play the load run';
        if (i === levels.length - 1 && animate) loadResult(levels[i]);
    }

    function loadResult(c) {
        const { load } = loadData();
        const bf = pointAt(load.FP16 || [], c), q = pointAt(load.INT4 || [], c), sp = pointAt(load.SPEC_DECODE || [], c);
        const parts = [];
        if (bf && q) parts.push(`With ${c} requests at once, ${esc(label('INT4'))} serves <b>${(q.output_tokens_per_second_per_gpu / bf.output_tokens_per_second_per_gpu).toFixed(1)}× the tokens per GPU</b> of ${esc(label('FP16'))}`);
        if (bf && sp && bf.latency_ms && sp.latency_ms) {
            const r = sp.latency_ms / bf.latency_ms;
            const verdict = Math.abs(r - 1) < 0.03 ? '<b>at about the same speed</b> as'
                : r < 1 ? `<b>${Math.round(100 * (1 - r))}% faster</b> than` : `<b>${Math.round(100 * (r - 1))}% slower</b> than`;
            parts.push(`${esc(label('SPEC_DECODE'))} answers ${verdict} ${esc(label('FP16'))} on the same GPUs`);
        }
        $('#loadResult').innerHTML = parts.length ? parts.join(', and ') + '.' : '';
    }

    function playLoad() {
        const { levels } = loadData();
        if (!levels.length) return;
        clearTimeout(loadTimer);
        $('#loadResult').innerHTML = '';
        for (const el of $$('#loadStage .num')) el.dataset.v = '0';
        let i = 0;
        const next = () => {
            showLevel(i, true);
            i += 1;
            if (i < levels.length) loadTimer = setTimeout(next, 2200);
        };
        next();
    }
    $('#loadPlay').addEventListener('click', () => { $('#loadPlay').blur(); playLoad(); });

    // ------------------------------------------------------------ boot

    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            if (!res.ok) throw new Error(res.status);
            config = await res.json();
        } catch (e) { config = null; }
        const badge = $('#modeBadge');
        if (config) {
            const live = config.mode === 'live';
            badge.className = `mode ${live ? 'live' : 'sim'}`;
            $('span', badge).textContent = live ? 'Live models' : 'Replay';
            badge.title = live ? 'Answers come from the vLLM deployments' : 'Models not connected, so each column replays the speed measured on the H200s';
        } else {
            badge.className = 'mode';
            $('span', badge).textContent = 'Offline';
        }
        buildAsk();
        buildLoad();
        buildNumbers();
        setOrder();
    }

    document.addEventListener('keydown', e => {
        const typing = e.target.closest('input, textarea');
        if (e.key === 'Escape' && typing) { e.target.blur(); return; }
        if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
        switch (e.code) {
            case 'Digit1': case 'Digit2': case 'Digit3': {
                const scene = order[Number(e.code.slice(5)) - 1];
                if (scene) show(scene);
                break;
            }
            case 'Space': if ($('#scene-load').classList.contains('is-active')) { e.preventDefault(); playLoad(); } break;
            case 'Slash': e.preventDefault(); show('ask'); $('#askInput').focus(); break;
            case 'KeyT': setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); break;
            case 'KeyF':
                if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
                break;
            default: break;
        }
    });

    fit();
    loadConfig();
})();
