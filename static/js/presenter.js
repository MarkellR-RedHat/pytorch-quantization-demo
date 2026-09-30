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
            gets: 'Half the GPUs, and about the same output per GPU under load',
            best: 'Everyday chat and easy questions',
            watch: '3 to 4 points lower on 280 MMLU-Pro questions, too few to call it, so the hardest questions stay on BF16 until it\'s tested further',
            route: 'Everyday questions',
        },
        SPEC_DECODE: {
            color: '--v-spec', role: 'Same 2 GPUs, 70B + 8B draft',
            gets: 'Faster answers on the same GPUs, with BF16 quality',
            best: 'Latency-sensitive, low-traffic work',
            watch: 'A slower first token, and about half of BF16\'s tokens per GPU under load: it\'s a latency tool',
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

    // Numbers and Under load describe the benchmark data, so they use the label it was measured under.
    function benchLabel(key) {
        const b = bench(key);
        return (b && b.label) || label(key);
    }
    // "INT4 (LLM Compressor)" as just "INT4" where the sentence is about the setup, not the checkpoint
    const shortLabel = key => benchLabel(key).replace(/ \(.*\)$/, '');

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
        if (/\b(prove|step by step|how many|calculate|reason\w*|riddle|puzzle|logic|math)\b/.test(t)) {
            return { key: 'FP16', why: 'it\'s multi-step reasoning, where INT4 hasn\'t been tested on a benchmark suite yet' };
        }
        if (/\b(explain|describe|essay|report|story|function|code|script)\b|\b\d{3,} words\b/.test(t)) {
            return { key: 'SPEC_DECODE', why: 'it\'s a long answer with someone waiting, and Spec Decode answers fastest' };
        }
        return { key: 'INT4', why: 'nothing here needs the full model, so the cheapest tokens win' };
    }

    // ------------------------------------------------------------ ask

    function roleText(key) {
        const v = config && config.variants.find(x => x.key === key);
        const role = (SETUPS[key] || {}).role || '';
        return v && v.build ? `${role}, ${v.build}` : role;
    }

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
                    <p class="role">${esc(roleText(key))}<span class="src" hidden></span></p>
                </div>
                <pre class="answer idle">Waiting for a question.</pre>
                <div class="astats">
                    <div><b class="num s-ttft">–</b><span class="s-ttft-label">first token</span></div>
                    <div><b class="num s-tps">–</b><span class="s-tps-label">tokens/s</span></div>
                    <div><b class="num s-len">–</b><span>tokens in answer</span></div>
                    <div><b class="num s-total">–</b><span class="s-total-label">total</span></div>
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
        let raw = '', live = '', recorded = false;
        const render = () => {
            if (!recorded) { out.textContent = markdownToText(raw); return; }
            // the live tokens stay visible above the divider, the recording goes below it
            out.replaceChildren();
            if (live) out.append(Object.assign(document.createElement('span'), { className: 'live-part', textContent: markdownToText(live) }));
            out.append(Object.assign(document.createElement('span'), { className: 'fb-divider', textContent: recorded }));
            out.append(Object.assign(document.createElement('span'), { className: 'rec-part', textContent: markdownToText(raw) }));
        };
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
                        src.classList.remove('recorded', 'planned');
                        out.classList.remove('planned');
                    } else if (ev.t === 'fallback' || ev.t === 'recorded') {
                        // 'fallback' = a live request failed; 'recorded' = this setup is recorded by plan today
                        live = raw;
                        raw = '';
                        recorded = ev.label;
                        out.classList.toggle('planned', ev.t === 'recorded');
                        const src = $('.src', col);
                        src.hidden = false;
                        src.textContent = 'recorded';
                        src.classList.toggle('recorded', ev.t === 'fallback');
                        src.classList.toggle('planned', ev.t === 'recorded');
                        render();
                    } else if (ev.t === 'delta') {
                        raw += ev.text;
                        render();
                        out.scrollTop = out.scrollHeight;
                    } else if (ev.t === 'done') {
                        clearInterval(tick);
                        if (ev.note) out.classList.add('idle');
                        // Replay shows the Sep 29 benchmark's average first-token time (BF16 and INT4 only). A live
                        // first token goes over the VPN or a port-forward, so it isn't comparable to the in-pod benchmark.
                        const missing = ev.source === 'recorded' ? 'not recorded' : 'live only';
                        const ttft = $('.s-ttft', col);
                        ttft.classList.toggle('na', ev.ttft_ms == null);
                        ttft.textContent = ev.ttft_ms != null ? secs(ev.ttft_ms) : missing;
                        // every stat says where it came from: this live request, the recording, or the benchmark
                        const basis = { recorded: 'recorded', benchmark: 'benchmark' };
                        $('.s-ttft-label', col).textContent = ev.ttft_ms == null ? 'first token'
                            : ev.source === 'live' ? 'TTFT + network' : `${basis[ev.ttft_basis] || 'benchmark'} TTFT`;
                        $('.s-tps', col).textContent = ev.tokens_per_second != null ? ev.tokens_per_second.toFixed(0) : '–';
                        $('.s-tps-label', col).textContent = ev.source === 'live' ? 'tokens/s' : `${basis[ev.tps_basis] || 'benchmark'} tok/s`;
                        const total = $('.s-total', col);
                        total.classList.toggle('na', ev.total_ms == null);
                        total.textContent = ev.total_ms != null ? secs(ev.total_ms) : missing;
                        $('.s-total-label', col).textContent = ev.source !== 'live' && ev.total_ms != null ? 'recorded total' : 'total';
                        const len = $('.s-len', col);
                        len.classList.toggle('na', ev.completion_tokens == null);
                        len.textContent = ev.completion_tokens != null ? ev.completion_tokens : missing;
                        if (ev.source === 'live') {
                            finished += 1;
                            place.textContent = ['1st', '2nd', '3rd', '4th'][finished - 1] || '';
                            if (finished === 1) place.classList.add('first');
                        }
                    } else if (ev.t === 'error') {
                        throw new Error(ev.detail && ev.detail.startsWith('Live request failed') ? ev.detail : 'This model didn\'t answer.');
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
    // A long question shrinks to fit the box instead of running off its right edge on the big screen.
    function fitInput() {
        const input = $('#askInput');
        input.style.fontSize = '';
        let size = parseFloat(getComputedStyle(input).fontSize);
        while (input.scrollWidth > input.clientWidth && size > 15) {
            size -= 1;
            input.style.fontSize = `${size}px`;
        }
    }
    $('#askInput').addEventListener('input', fitInput);

    function buildPresets() {
        const box = $('#presets');
        $$('button', box).forEach(b => b.remove());
        for (const p of (config && config.presets) || []) {
            const b = Object.assign(document.createElement('button'), { type: 'button', textContent: p.label });
            b.dataset.preset = p.key;
            b.addEventListener('click', () => {
                b.blur();
                $('#askInput').value = p.prompt;
                fitInput();
                ask(p.prompt, p.key);
            });
            box.append(b);
        }
    }

    // ------------------------------------------------------------ numbers (the money slide)

    // The "gets" and "watch out" lines carry ratios computed from the benchmark, so the words never drift
    // from the numbers under them.
    function getsText(key, s, b, bf) {
        if (key === 'SPEC_DECODE' && b && bf && b.throughput_tps && bf.throughput_tps) {
            return `About ${(b.throughput_tps / bf.throughput_tps).toFixed(2)}× faster answers on the same GPUs, with BF16 quality`;
        }
        return s.gets || '';
    }
    function watchText(key, s, b, bf) {
        if (key === 'SPEC_DECODE' && b && bf && b.ttft_ms_avg && bf.ttft_ms_avg) {
            const t = targetInfo(), sb = t.best.SPEC_DECODE, bb = t.best.FP16;
            const share = sb && bb ? `about ${Math.round(100 * sb.output_tokens_per_second_per_gpu / bb.output_tokens_per_second_per_gpu)}%` : 'about half';
            return `First token about ${(b.ttft_ms_avg / bf.ttft_ms_avg).toFixed(0)}× slower than BF16's, and ${share} of BF16's tokens per GPU under load: it's a latency tool`;
        }
        return s.watch || '';
    }

    function buildNumbers() {
        const keys = variants();
        const grid = $('#moneyGrid');
        grid.style.setProperty('--cols', keys.length);
        grid.innerHTML = '';
        const bf = bench('FP16');
        for (const key of keys) {
            const s = SETUPS[key] || {}, b = bench(key), n = gpus(key);
            const tps = b ? b.throughput_tps : null;
            const mean = b ? b.avg_latency_ms : null;
            const gsm = b && b.accuracy && b.accuracy.gsm8k;
            // accuracy: the GSM8K score with its count (the reference, BF16, gets the same treatment);
            // MMLU-Pro is in the Watch out line and the footnote, with its count and error
            const acc = s.acc ? { text: s.acc, note: s.accNote, small: false }
                : gsm ? { text: `${gsm.score.toFixed(1)}%`, note: `GSM8K, ${gsm.questions.toLocaleString('en-US')} questions`, small: false }
                : { text: 'Pending', note: 'suite eval pending', small: true };
            const card = document.createElement('div');
            card.className = 'card mcard';
            card.style.setProperty('--c', `var(${color(key)})`);
            card.innerHTML = `
                <div class="mhead"><h3>${glyph(key)}${esc(benchLabel(key))}</h3>${chips(key)}</div>
                <p class="gets">${esc(getsText(key, s, b, bf))}</p>
                <div class="nums three">
                    <div><b class="num">${tps != null ? one(tps) : '–'}</b><span>tokens/s, one request</span></div>
                    <div><b class="num">${mean != null ? `${(mean / 1000).toFixed(2)} s` : '–'}</b><span>mean time, one request</span></div>
                    <div><b class="num${acc.small ? ' small' : ''}">${esc(acc.text)}</b><span>${esc(acc.note)}</span></div>
                </div>
                <dl><div><dt>Best for</dt><dd>${esc(s.best || '')}</dd></div>${s.watch ? `<div><dt>Watch out for</dt><dd>${esc(watchText(key, s, b, bf))}</dd></div>` : ''}</dl>`;
            grid.appendChild(card);
        }
        const q = bench('INT4');
        if (bf && q && bf.throughput_tps && q.throughput_tps) {
            const speed = q.throughput_tps / bf.throughput_tps;
            const sp = bench('SPEC_DECODE');
            const spTxt = sp && sp.throughput_tps ? `, and ${esc(shortLabel('SPEC_DECODE'))} <b>about ${(sp.throughput_tps / bf.throughput_tps).toFixed(2)}× faster</b> on the same GPUs` : '';
            $('#takeaway').innerHTML = `One request at a time, ${esc(shortLabel('INT4'))} runs at <b>${Math.round(100 * speed)}% of ${esc(shortLabel('FP16'))}'s speed on half the GPUs</b>${spTxt}.`
                + (loadData().keys.length ? costLine() : ' <span class="pending-note">Under heavy batching on high-end GPUs, Red Hat\'s study found 8-bit (W8A8) more cost-efficient than 4-bit, and the load test shows where these three land.</span>');
        }
        const strip = $('#routerStrip');
        strip.className = 'card router-strip';
        strip.innerHTML = `<h3>Big, mixed traffic? Route it</h3>` + ['FP16', 'INT4', 'SPEC_DECODE'].filter(k => keys.includes(k)).map(k =>
            // the lane names the setup, not the checkpoint, so "INT4 (LLM Compressor)" shows as INT4 here
            `<div class="route">${esc(SETUPS[k].route)} <span class="arrow">→</span> ${glyph(k)}<b>${esc(label(k).replace(/ \(.*\)$/, ''))}</b></div>`).join('')
            // the same lane idea combined, as on the router slide: not measured here
            + '<div class="route untested"><b>INT4 + spec decode</b><em>untested</em></div>';
        const bm = (config && config.benchmark) || {};
        const when = bm.date ? new Date(bm.date + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '';
        const sd = bench('SPEC_DECODE') || {};
        $('#numbersFoot').textContent = accuracyFoot() + specFoot(sd);
        $('#routerNote').textContent = 'A router pays off once your traffic is big and mixed enough to run more than one pool. With small traffic, pick the one setup that fits most of your questions.';
        const ss = bm.single_stream || {};
        const q4 = bench('INT4') || {};
        $('#benchNote').textContent = `Llama 3.1 70B Instruct on NVIDIA H200, vLLM ${bm.vllm_version || ''}, measured ${when} inside the pods with vllm bench serve: ${ss.prompts || 30} ${ss.dataset || 'ShareGPT'} prompts per setup, one at a time, temperature 0.`
            + (q4.build ? ` INT4 is ${q4.build.replace('Red Hat LLM Compressor build', 'Red Hat\'s LLM Compressor build')}${q4.kernel ? ` on vLLM's ${q4.kernel.replace('LinearKernel', '')} kernel` : ''}.` : '')
            + (keys.includes('FP8') ? '' : ' FP8 (one H200, 99.9% of BF16 on OpenLLM v1 in Red Hat\'s tests) is the next to measure.');
    }

    // The accuracy footnote, with the sample sizes and the standard error, so the claim is exactly as
    // strong as the data: GSM8K on all 1,319 questions, MMLU-Pro on 280.
    function accuracyFoot() {
        const bf = bench('FP16'), q = bench('INT4');
        const a = bf && bf.accuracy, qa = q && q.accuracy;
        if (!a || !qa) return '';
        const ref = q.reference;
        const g = a.gsm8k, qg = qa.gsm8k, m = a.mmlu_pro, qm = qa.mmlu_pro;
        const pts = m.score - qm.score;
        const naive = ref ? `The naive pick, the ${ref.build.replace(', the naive pick', '')}: ${one(ref.throughput_tps)} tokens/s (${Math.round(100 * ref.speed_vs_baseline)}% of BF16), GSM8K ${ref.gsm8k.toFixed(1)}%, MMLU-Pro ${ref.mmlu_pro.toFixed(1)}%. ` : '';
        return naive + `GSM8K (${g.fewshot}-shot chain of thought, all ${g.questions.toLocaleString('en-US')} questions): BF16 ${g.score.toFixed(1)}%, INT4 ${qg.score.toFixed(1)}%: no loss. MMLU-Pro (${m.fewshot}-shot, the first ${m.per_subject} questions of ${m.subjects} subjects, ${m.questions} in all, standard error about ±${m.stderr.toFixed(1)} points): BF16 ${m.score.toFixed(1)}%, INT4 ${qm.score.toFixed(1)}%: ${Math.floor(pts)} to ${Math.ceil(pts)} points lower on ${m.questions} questions, too few to call it, so the hard questions stay on BF16 until it's tested further. `;
    }
    function specFoot(sd) {
        const acc = sd.acceptance;
        if (!acc) return `Spec Decode used Llama 3.1 8B as the draft, proposing 5 tokens per step${sd.mean_acceptance_length ? `, and averaged ${one(sd.mean_acceptance_length)} tokens per 70B pass.` : '.'}`;
        const a0 = acc['0'], a7 = acc['0.7'];
        return `Spec Decode's 8B draft proposes 5 tokens per step: ${(100 * a0.rate).toFixed(1)}% accepted at temperature 0 (${(100 * a7.rate).toFixed(1)}% at 0.7), ${one(a0.mean_acceptance_length)} tokens per 70B pass (${one(a7.mean_acceptance_length)}), over ${a0.drafts.toLocaleString('en-US')} and ${a7.drafts.toLocaleString('en-US')} drafts.`;
    }

    // The cost line: output tokens per GPU at the latency target, from the load test, and a price
    // per million output tokens only when GPU_HOURLY_USD is set (it's an assumption, and says so).
    function costLine() {
        const t = targetInfo();
        const keys = variants().filter(k => t.best[k]);
        if (!keys.length) return '';
        const usd = config && config.gpu_hourly_usd;
        const items = keys.map(k => {
            const tps = t.best[k].output_tokens_per_second_per_gpu;
            const price = usd > 0 ? ` ($${(usd / (tps * 3600) * 1e6).toFixed(2)} per 1M tokens)` : '';
            return `${esc(shortLabel(k))} ${tps.toFixed(0)}${price}`;
        });
        const assumed = usd > 0 ? `, at an assumed $${usd}/GPU-hour` : '';
        return ` <span class="pending-note">Under load, output tokens/s per GPU while the ${t.kind || 'tail'} time per token stays under ${t.ms} ms${assumed}: ${items.join(' · ')}.</span>`;
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

    // The latency target the throughput has to be earned under, and the tail it's measured at.
    function targetInfo() {
        const bm = (config && config.benchmark) || {};
        const load = bm.load || {};
        const kinds = new Set(Object.values(load).flat().map(p => p.tpot_tail_kind).filter(Boolean));
        const kind = kinds.size === 1 ? [...kinds][0] : kinds.size ? 'tail' : null;
        return { ms: bm.tpot_target_ms || 50, kind, best: bm.at_target || {} };
    }
    const perToken = t => `${t.kind || 'tail'} time per output token`;

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
                <div class="mhead"><h3>${glyph(key)}${esc(benchLabel(key))}</h3>${chips(key)}</div>
                <div class="big"><b class="num l-pergpu">–</b><span>output tokens/s per GPU</span></div>
                <div class="spark-box"><svg class="spark"></svg></div>
                <div class="tpot-wrap"><span class="tpot-title"></span><svg class="tpot"></svg></div>
                <div class="lstats">
                    <div><b class="num l-total">–</b><span>tokens/s, whole setup</span></div>
                    <div><b class="num l-tpot">–</b><span class="l-tpot-label"></span></div>
                    <div><b class="num l-lat">–</b><span class="l-lat-label">median time per answer</span></div>
                </div>
                <p class="l-target"></p>`;
            card.dataset.max = maxPerGpu;
            stage.appendChild(card);
            const t = targetInfo();
            $('.tpot-title', card).textContent = `${perToken(t)}, with the ${t.ms} ms target`;
            $('.l-tpot-label', card).textContent = perToken(t);
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
                if (p.tpot_tail_ms != null) tween($('.l-tpot', card), p.tpot_tail_ms, v => `${v.toFixed(0)} ms`, ms);
                $('.l-lat-label', card).textContent = `${p.latency_kind === 'mean' ? 'mean' : 'median'} time per answer`;
            }
            drawSpark(card, pts, levels, i);
            drawTpot(card, pts, levels, i);
            const t = targetInfo(), best = t.best[card.dataset.key];
            $('.l-target', card).innerHTML = i < levels.length - 1 ? '' : best
                ? `Under the ${t.ms} ms target: <b>${best.output_tokens_per_second_per_gpu.toFixed(0)} tokens/s per GPU</b> at ${best.concurrency} in flight`
                : `Never stayed under the ${t.ms} ms target`;
        }
        const play = $('#loadPlay');
        play.textContent = animate && i < levels.length - 1 ? `${levels[i]} at a time…` : 'Play the load run';
        if (i === levels.length - 1 && animate) loadResult();
        // redraw once the text around the charts has settled, so each chart measures its final box
        requestAnimationFrame(() => {
            for (const card of $$('#loadStage .lcard')) {
                drawSpark(card, load[card.dataset.key], levels, i);
                drawTpot(card, load[card.dataset.key], levels, i);
            }
        });
    }

    function drawSpark(card, pts, levels, i) {
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

    // tail time per output token against concurrency, with the latency target as a dashed line
    function drawTpot(card, pts, levels, i) {
        const el = $('.tpot', card), t = targetInfo();
        const W = Math.max(200, el.clientWidth), H = Math.max(60, el.clientHeight);
        el.setAttribute('viewBox', `0 0 ${W} ${H}`);
        const all = (config.benchmark.load ? Object.values(config.benchmark.load).flat() : []).map(p => p.tpot_tail_ms).filter(v => v != null);
        const max = Math.max(t.ms * 1.25, ...all) * 1.08, L = 12, B = 6;
        const x = j => L + j * (W - 2 * L) / Math.max(1, levels.length - 1);
        const y = v => 4 + (1 - v / max) * (H - B - 4);
        const shown = levels.slice(0, i + 1).map((c, j) => ({ j, p: pointAt(pts, c) })).filter(o => o.p && o.p.tpot_tail_ms != null);
        let svg = `<line class="target" x1="${L}" x2="${W - L}" y1="${y(t.ms)}" y2="${y(t.ms)}"/>`;
        svg += `<text class="target-label" x="${L}" y="${y(t.ms) - 7}">${t.ms} ms target</text>`;
        if (shown.length > 1) svg += `<polyline points="${shown.map(o => `${x(o.j)},${y(o.p.tpot_tail_ms)}`).join(' ')}"/>`;
        svg += shown.map(o => `<circle class="${o.p.tpot_tail_ms > t.ms ? 'over' : ''}" cx="${x(o.j)}" cy="${y(o.p.tpot_tail_ms)}" r="6"/>`).join('');
        el.innerHTML = svg;
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
            parts.push(`At ${perGpu} requests per GPU, ${esc(benchLabel('INT4'))} serves <b>${one(r)}× the output tokens per GPU</b> of ${esc(benchLabel('FP16'))}`);
        }
        const same = [...levels].reverse().map(c => ({ c, bf: pointAt(bfPts, c), sp: pointAt(spPts, c) })).find(o => o.bf && o.sp && o.bf.latency_ms && o.sp.latency_ms);
        if (same) {
            const r = same.sp.latency_ms / same.bf.latency_ms;
            const kind = same.bf.latency_kind === 'mean' ? 'mean' : 'median';
            const verdict = Math.abs(r - 1) < 0.03 ? 'about the same as' : r < 1 ? `<b>${Math.round(100 * (1 - r))}% lower</b> than` : `<b>${Math.round(100 * (r - 1))}% higher</b> than`;
            parts.push(`at ${same.c} requests on the same 2 GPUs, ${esc(benchLabel('SPEC_DECODE'))}'s ${kind} time per answer is ${verdict} ${esc(benchLabel('FP16'))}'s`);
        }
        // what sets cost: output tokens per GPU while the tail time per token stays under the target
        const t = targetInfo(), bfBest = t.best.FP16;
        const cost = ['INT4', 'SPEC_DECODE'].filter(k => bfBest && t.best[k]).map(k =>
            `${esc(shortLabel(k))} <b>${one(t.best[k].output_tokens_per_second_per_gpu / bfBest.output_tokens_per_second_per_gpu)}×</b>`);
        if (cost.length) {
            // with a latency target, the per-GPU comparison at the target replaces the equal-load one
            const lead = `Within a ${t.ms} ms ${t.kind || 'tail'} time per token, output tokens per GPU against ${esc(shortLabel('FP16'))}: ${cost.join(', ')}.`;
            $('#loadResult').innerHTML = lead + (same ? ` ${parts[parts.length - 1].replace(/^at/, 'At')}.` : '');
            return;
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

    // The corner badge says exactly which setups are live and which are playing recordings.
    function badgeState(cfg) {
        if (cfg.mode !== 'live') return { kind: 'sim', text: 'Replay' };
        const recorded = cfg.variants.filter(v => v.mode === 'recorded');
        if (!recorded.length) return { kind: 'live', text: 'Live models' };
        const names = list => list.map(v => v.label).join(', ');
        const live = cfg.variants.filter(v => v.mode !== 'recorded');
        return { kind: 'mixed', text: `${live.length ? `Live: ${names(live)} · ` : ''}Recorded: ${names(recorded)}` };
    }

    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            if (!res.ok) throw new Error(res.status);
            config = await res.json();
        } catch (e) { config = null; }
        const badge = $('#modeBadge');
        if (config) {
            const live = config.mode === 'live';
            const b = badgeState(config);
            badge.className = `mode ${b.kind}`;
            $('span', badge).textContent = b.text;
            badge.title = live ? 'Answers come from the vLLM deployments, except any setup marked recorded' : 'Models not connected, so each column replays the speed measured on the H200s';
        } else {
            badge.className = 'mode';
            $('span', badge).textContent = 'Offline';
        }
        buildPresets();
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
            case 'KeyR': toggleReplay(); break;
            case 'KeyT': setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); break;
            case 'KeyF':
                if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => {});
                break;
            default: break;
        }
    });

    // R flips every column between the live models and replay in one keystroke, for when the network dies.
    async function toggleReplay() {
        try {
            const res = await fetch('/simulation/toggle', { method: 'POST', credentials: 'same-origin' });
            if (!res.ok) throw new Error(res.status);
        } catch (e) { return; }
        running.forEach(c => c.abort());
        running = [];
        await loadConfig();
    }

    fit();
    loadConfig();
})();
