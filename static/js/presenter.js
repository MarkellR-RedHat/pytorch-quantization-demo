(function () {
    'use strict';
    const $ = (s, el = document) => el.querySelector(s);
    const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

    // What each setup buys you. Wording matches the slides.
    const SETUPS = {
        FP16: {
            color: '--v-bf16', role: 'The reference',
            gets: 'The reference the other two are measured against',
            best: 'Your hardest questions, until INT4 is tested on them',
            watch: 'Every replica needs two GPUs',
            route: 'A wrong answer is expensive',
            acc: '100%', accNote: 'the reference',
        },
        INT4: {
            color: '--v-int4', role: 'Half the GPUs',
            gets: 'Half the GPUs, about the same speed up to 8 requests at once',
            best: 'Everyday chat and easy questions',
            watch: 'Not tested on a benchmark suite yet, so test it on your own prompts',
            route: 'Everyday questions',
            acc: 'Pending', accSmall: true, accNote: 'suite eval pending*',
        },
        SPEC_DECODE: {
            color: '--v-spec', role: 'Same 2 GPUs, 70B + 8B draft',
            gets: 'About 1.4× faster answers on the same GPUs, with BF16 quality',
            best: 'Latency-sensitive, low-traffic work',
            watch: 'The draft model leaves less room for KV cache, and it isn\'t load-tested yet',
            route: 'Latency-sensitive, low traffic',
            acc: '= BF16', accNote: 'by design, the 70B checks every token',
        },
        FP8: {
            color: '--v-fp8', role: 'One GPU, 8-bit',
            gets: '8-bit weights and activations on a single GPU',
            best: 'High-throughput serving on Hopper GPUs',
            watch: 'Not measured in this run yet',
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
        if (/\b(legal|contract|diagnos\w*|compliance|medical|financial|audit)\b/.test(t)) {
            return { key: 'FP16', why: 'a wrong answer here is expensive' };
        }
        if (/\b(why|prove|step by step|how many|calculate|reason\w*|riddle|puzzle|math)\b/.test(t)) {
            return { key: 'FP16', why: 'it\'s multi-step reasoning, where INT4 hasn\'t been tested on a benchmark suite yet' };
        }
        if (/\b(write|draft|explain|describe|story|essay|report)\b/.test(t) && t.length > 60) {
            return { key: 'SPEC_DECODE', why: 'it\'s a long answer with someone waiting, and Spec Decode answers fastest' };
        }
        return { key: 'INT4', why: 'nothing here needs the full model, so the cheapest tokens win' };
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
                    <p class="role">${esc((SETUPS[key] || {}).role || '')}<span class="src" hidden></span></p>
                </div>
                <pre class="answer idle">Waiting for a question.</pre>
                <div class="astats">
                    <div><b class="num s-ttft">–</b><span class="s-ttft-label">first token</span></div>
                    <div><b class="num s-tps">–</b><span class="s-tps-label">tokens/s</span></div>
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
                    if (ev.t === 'start') {
                        const src = $('.src', col);
                        const tag = ev.source === 'replay' ? { captured: 'captured answer' }[ev.text_source] : '';
                        src.hidden = !tag;
                        src.textContent = tag || '';
                    } else if (ev.t === 'delta') {
                        raw += ev.text;
                        out.textContent = markdownToText(raw);
                        out.scrollTop = out.scrollHeight;
                    } else if (ev.t === 'done') {
                        clearInterval(tick);
                        // Replay shows the Sep 29 benchmark's average first-token time (BF16 and INT4 only).
                        const ttft = $('.s-ttft', col);
                        ttft.classList.toggle('na', ev.ttft_ms == null);
                        ttft.textContent = ev.ttft_ms != null ? secs(ev.ttft_ms) : 'live only';
                        $('.s-ttft-label', col).textContent = ev.source === 'replay' && ev.ttft_ms != null ? 'benchmark TTFT' : 'first token';
                        $('.s-tps', col).textContent = ev.tokens_per_second != null ? ev.tokens_per_second.toFixed(0) : '–';
                        $('.s-tps-label', col).textContent = ev.source === 'replay' ? 'benchmark tok/s' : 'tokens/s';
                        const total = $('.s-total', col);
                        total.classList.toggle('na', ev.total_ms == null);
                        total.textContent = ev.total_ms != null ? secs(ev.total_ms) : 'live only';
                        const len = $('.s-len', col);
                        len.classList.toggle('na', ev.completion_tokens == null);
                        len.textContent = ev.completion_tokens != null ? ev.completion_tokens : 'live only';
                        if (ev.source === 'live') {
                            finished += 1;
                            place.textContent = ['1st', '2nd', '3rd', '4th'][finished - 1] || '';
                            if (finished === 1) place.classList.add('first');
                        }
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
            const mean = b ? b.avg_latency_ms : null;
            const card = document.createElement('div');
            card.className = 'card mcard';
            card.style.setProperty('--c', `var(${color(key)})`);
            card.innerHTML = `
                <div class="mhead"><h3>${glyph(key)}${esc(label(key))}</h3>${chips(key)}</div>
                <p class="gets">${esc(s.gets || '')}</p>
                <div class="nums three">
                    <div><b class="num">${tps != null ? one(tps) : '–'}</b><span>tokens/s, one request</span></div>
                    <div><b class="num">${mean != null ? `${(mean / 1000).toFixed(2)} s` : '–'}</b><span>mean time, one request</span></div>
                    <div><b class="num${s.accSmall ? ' small' : ''}">${esc(s.acc || '–')}</b><span>${esc(s.accNote || 'accuracy vs BF16')}</span></div>
                </div>
                <dl><div><dt>Best for</dt><dd>${esc(s.best || '')}</dd></div>${s.watch ? `<div><dt>Watch out for</dt><dd>${esc(s.watch)}</dd></div>` : ''}</dl>`;
            grid.appendChild(card);
        }
        const bf = bench('FP16'), q = bench('INT4');
        if (bf && q && bf.throughput_tps && q.throughput_tps) {
            const speed = q.throughput_tps / bf.throughput_tps;
            const sp = bench('SPEC_DECODE');
            const spTxt = sp && sp.throughput_tps ? `, and ${esc(label('SPEC_DECODE'))} runs <b>about ${one(sp.throughput_tps / bf.throughput_tps)}× faster</b> on the same GPUs` : '';
            $('#takeaway').innerHTML = `One request at a time, ${esc(label('INT4'))} runs at <b>${Math.round(100 * speed)}% of ${esc(label('FP16'))}'s speed on half the GPUs</b>${spTxt}.`
                + (loadData().keys.length ? '' : ' <span class="pending-note">Under heavy batching on high-end GPUs, Red Hat\'s study found 8-bit (W8A8) more cost-efficient than 4-bit, and the load test shows where these three land.</span>');
        }
        const strip = $('#routerStrip');
        strip.className = 'card router-strip';
        strip.innerHTML = `<h3>Big, mixed traffic? Route it</h3>` + ['FP16', 'INT4', 'SPEC_DECODE'].filter(k => keys.includes(k)).map(k =>
            `<div class="route">${esc(SETUPS[k].route)} <span class="arrow">→</span> ${glyph(k)}<b>${esc(label(k))}</b></div>`).join('');
        const bm = (config && config.benchmark) || {};
        const when = bm.date ? new Date(bm.date + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '';
        const sd = bench('SPEC_DECODE') || {};
        const mal = sd.mean_acceptance_length || null;
        $('#numbersFoot').textContent = '* All three setups answered the sheep riddle correctly 20 out of 20 times at temperature 0.7, but the INT4 build here (hugging-quants AWQ, W4A16 on vLLM\'s Machete kernel) hasn\'t been run on a benchmark suite yet. Red Hat\'s published INT4 build of this model (GPTQ) recovers 99.4% of BF16 on OpenLLM v1 and 97.4% on the harder v2 set. Spec Decode used Llama 3.1 8B as the draft, proposing 5 tokens per step'
            + (mal ? `, and averaged ${one(mal)} tokens per 70B pass.` : '.');
        $('#routerNote').textContent = 'A router pays off once your traffic is big and mixed enough to run more than one pool. With small traffic, pick the one setup that fits most of your questions.';
        $('#benchNote').textContent = `Llama 3.1 70B Instruct on NVIDIA H200 with vLLM ${bm.vllm_version || ''}, measured ${when}: 5 runs per setup, one request at a time, temperature 0, 256 output tokens, enforce_eager off everywhere. Spec Decode's average includes one cold run.`
            + (keys.includes('FP8') ? '' : ' FP8 fits on one H200 and recovers 99.9% of BF16 on OpenLLM v1 in Red Hat\'s published tests, so it\'s the next setup to measure.');
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
        const sizes = any ? ` Synthetic random-token prompts of about ${any.avg_input_tokens} tokens, each asking for ${any.avg_output_tokens}.` : ' Synthetic random-token prompts.';
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
                <div class="big"><b class="num l-pergpu">–</b><span>output tokens/s per GPU</span></div>
                <svg class="spark"></svg>
                <div class="lstats">
                    <div><b class="num l-total">–</b><span>output tokens/s, whole setup</span></div>
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
        if (i === levels.length - 1 && animate) loadResult();
    }

    // Compare INT4 with BF16 at the same load per GPU (INT4 runs on 1 GPU, BF16 on 2), and
    // Spec Decode with BF16 at the same load, since they run on the same 2 GPUs.
    function loadResult() {
        const { load, levels } = loadData();
        const bfPts = load.FP16 || [], qPts = load.INT4 || [], spPts = load.SPEC_DECODE || [];
        const parts = [];
        const gb = gpus('FP16'), gq = gpus('INT4');
        const pair = [...levels].reverse().map(c => ({ c, bf: pointAt(bfPts, c), q: pointAt(qPts, c * gq / gb) })).find(o => o.bf && o.q);
        if (pair) {
            const perGpu = pair.c / gb;
            const r = pair.q.output_tokens_per_second_per_gpu / pair.bf.output_tokens_per_second_per_gpu;
            parts.push(`At ${perGpu} requests per GPU, ${esc(label('INT4'))} serves <b>${one(r)}× the output tokens per GPU</b> of ${esc(label('FP16'))}`);
        }
        const same = [...levels].reverse().map(c => ({ c, bf: pointAt(bfPts, c), sp: pointAt(spPts, c) })).find(o => o.bf && o.sp && o.bf.latency_ms && o.sp.latency_ms);
        if (same) {
            const r = same.sp.latency_ms / same.bf.latency_ms;
            const kind = same.bf.latency_kind === 'mean' ? 'mean' : 'median';
            const verdict = Math.abs(r - 1) < 0.03 ? 'about the same as' : r < 1 ? `<b>${Math.round(100 * (1 - r))}% lower</b> than` : `<b>${Math.round(100 * (r - 1))}% higher</b> than`;
            parts.push(`at ${same.c} requests on the same 2 GPUs, ${esc(label('SPEC_DECODE'))}'s ${kind} time per answer is ${verdict} ${esc(label('FP16'))}'s`);
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
