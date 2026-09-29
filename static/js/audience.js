(function () {
    'use strict';
    const $ = (s, el = document) => el.querySelector(s);
    const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

    const BIRDS = [
        { key: 'BF16', name: 'BF16', color: '--v-bf16' },
        { key: 'FP8', name: 'FP8', color: '--v-fp8' },
        { key: 'INT4_AWQ', name: 'INT4 AWQ', color: '--v-int4' },
        { key: 'INT4_RTN', name: 'INT4 RTN', color: '--v-int4', hollow: true },
        { key: 'SPEC', name: 'Spec Decode', color: '--v-spec' },
    ];
    const VARIANT_COLOR = { FP16: '--v-bf16', FP8: '--v-fp8', INT4: '--v-int4', SPEC_DECODE: '--v-spec' };
    const COOLDOWN_MS = 4000;
    const MAX_PENDING = 3;

    let backed = null;
    let arenaState = null;
    let votes = {};
    let variants = [];
    let selected = null;
    let pending = 0;
    let sent = 0;

    function storeGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
    function storeSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }

    let toastTimer = null;
    function toast(text) {
        const t = $('#toast');
        t.textContent = text;
        t.classList.add('show');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
    }

    // ------------------------------------------------------------ back a bird

    function buildBirds() {
        const box = $('#birds');
        box.innerHTML = '';
        for (const b of BIRDS) {
            const btn = document.createElement('button');
            btn.className = 'bird';
            btn.setAttribute('role', 'radio');
            btn.setAttribute('aria-checked', String(backed === b.key));
            btn.dataset.key = b.key;
            btn.style.setProperty('--c', `var(${b.color})`);
            btn.innerHTML = `<span class="glyph${b.hollow ? ' hollow' : ''}" style="--c: var(${b.color})"></span><b></b><span class="pct num"></span>`;
            $('b', btn).textContent = b.name;
            btn.addEventListener('click', () => back(b.key));
            box.appendChild(btn);
        }
        renderBirds();
    }

    async function back(key) {
        backed = key;
        storeSet('qs-backed', key);
        renderBirds();
        try {
            const res = await fetch('/arena/back', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ variant: key }) });
            if (res.ok) { const d = await res.json(); votes = d.counts || {}; renderBirds(); }
            else if (res.status === 429) toast('Easy there, one change at a time.');
        } catch (e) { toast('Could not reach the server.'); }
    }

    function hardPct(key) {
        const b = arenaState && arenaState.birds && arenaState.birds[key];
        if (!b || !b.hard || !b.hard[1]) return null;
        return Math.round(100 * b.hard[0] / b.hard[1]);
    }

    function renderBirds() {
        for (const btn of $$('#birds .bird')) {
            const key = btn.dataset.key;
            btn.setAttribute('aria-checked', String(backed === key));
            const pct = hardPct(key);
            $('.pct', btn).textContent = pct === null ? '–' : `${pct}%`;
        }
        const total = Object.values(votes).reduce((a, b) => a + b, 0);
        $('#backHint').textContent = total ? `${total} backing · % = hard gaps cleared` : '% = hard gaps cleared';
        const y = $('#yours');
        if (!backed) { y.hidden = true; return; }
        const bird = BIRDS.find(b => b.key === backed);
        const s = arenaState && arenaState.birds && arenaState.birds[backed];
        y.hidden = false;
        if (!s) {
            y.textContent = `You're backing ${bird.name}. Stats show up once the presenter starts the arena.`;
            return;
        }
        const others = (votes[backed] || 1) - 1;
        y.innerHTML = '';
        y.append(`You're backing `);
        const b = document.createElement('b');
        b.textContent = bird.name;
        y.append(b, `, which has cleared ${s.hard[0]} of ${s.hard[1]} hard gaps with ${s.crashes} crash${s.crashes === 1 ? '' : 'es'} so far`);
        y.append(others > 0 ? `, and ${others} other ${others === 1 ? 'person is' : 'people are'} backing it too.` : '.');
    }

    // ------------------------------------------------------------ hard prompts

    function cooldown(ms) {
        const until = performance.now() + ms;
        $$('.throw').forEach(b => { b.disabled = true; });
        function step(t) {
            const left = Math.max(0, until - t);
            $$('.throw i').forEach(i => { i.style.width = `${100 * left / ms}%`; });
            if (left > 0) requestAnimationFrame(step);
            else $$('.throw').forEach(b => { b.disabled = false; });
        }
        requestAnimationFrame(step);
    }

    $$('.throw').forEach(btn => btn.addEventListener('click', async () => {
        if (btn.disabled) return;
        cooldown(COOLDOWN_MS);
        try {
            const res = await fetch('/arena/hard-prompt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: btn.dataset.kind }) });
            if (res.ok) {
                const d = await res.json();
                toast(`${d.label} is on its way. Watch the big screen.`);
            } else if (res.status === 429) {
                const retry = Number(res.headers.get('Retry-After')) || 2;
                toast('Lots of people throwing right now, try again in a second.');
                cooldown(retry * 1000);
            } else {
                toast('That one didn\'t go through.');
            }
        } catch (e) { toast('Could not reach the server.'); }
    }));

    // ------------------------------------------------------------ load test

    function buildVariants() {
        const seg = $('#variantSeg');
        seg.innerHTML = '';
        for (const v of variants) {
            const b = document.createElement('button');
            b.setAttribute('role', 'radio');
            b.setAttribute('aria-checked', String(v.key === selected));
            b.dataset.key = v.key;
            b.innerHTML = `<span class="glyph" style="--c: var(${VARIANT_COLOR[v.key] || '--ink-3'})"></span><span></span>`;
            $('span:last-child', b).textContent = { FP16: 'BF16', FP8: 'FP8', INT4: 'INT4', SPEC_DECODE: 'Spec' }[v.key] || v.label;
            b.addEventListener('click', () => {
                selected = v.key;
                storeSet('qs-variant', v.key);
                $$('#variantSeg button').forEach(x => x.setAttribute('aria-checked', String(x.dataset.key === selected)));
            });
            seg.appendChild(b);
        }
    }

    function addResult(label) {
        const li = document.createElement('li');
        li.className = 'pending';
        li.innerHTML = '<span class="v"></span><span class="s">waiting for the model…</span><span class="t num">…</span>';
        $('.v', li).textContent = label;
        const ol = $('#results');
        ol.prepend(li);
        while (ol.children.length > 3) ol.lastChild.remove();
        return li;
    }

    $('#sendBtn').addEventListener('click', async () => {
        if (!selected) return;
        if (pending >= MAX_PENDING) { toast('Three in flight already. Give them a second.'); return; }
        const v = variants.find(x => x.key === selected);
        const li = addResult(v ? v.label : selected);
        pending++;
        sent++;
        $('#yourCount').textContent = sent;
        $('#sendBtn').classList.toggle('busy', pending > 0);
        try {
            const res = await fetch('/request', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_type: selected, prompt_id: 'chat' }) });
            li.classList.remove('pending');
            if (res.ok) {
                const d = await res.json();
                $('.t', li).textContent = `${(d.latency_ms / 1000).toFixed(1)} s`;
                $('.s', li).textContent = `${d.completion_tokens} tokens · ${d.tokens_per_second.toFixed(0)} tok/s`;
            } else {
                li.classList.add('err');
                $('.t', li).textContent = res.status === 429 ? 'slow down' : res.status === 503 ? 'busy' : 'error';
                $('.s', li).textContent = res.status === 429 ? 'rate limited, try again' : res.status === 503 ? 'the GPUs are full right now' : 'the request failed';
            }
        } catch (e) {
            li.classList.remove('pending');
            li.classList.add('err');
            $('.t', li).textContent = 'offline';
            $('.s', li).textContent = 'could not reach the server';
        } finally {
            pending--;
            $('#sendBtn').classList.toggle('busy', pending > 0);
        }
    });

    // ------------------------------------------------------------ live updates

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${proto}//${location.host}/ws`);
        ws.onmessage = ev => {
            let msg;
            try { msg = JSON.parse(ev.data); } catch (e) { return; }
            const d = msg.data;
            if (msg.type === 'arena_state') { arenaState = d; renderBirds(); }
            else if (msg.type === 'arena_votes') { votes = (d && d.counts) || {}; renderBirds(); }
            else if (msg.type === 'state_update') {
                $('#peopleCount').textContent = d.participant_count ?? 0;
                $('#totalCount').textContent = (d.total_requests ?? 0).toLocaleString('en-US');
            }
        };
        ws.onclose = () => setTimeout(connect, 2000);
    }

    async function boot() {
        backed = storeGet('qs-backed');
        buildBirds();
        try {
            const [me, cfg] = await Promise.all([fetch('/api/me').then(r => r.json()), fetch('/api/config').then(r => r.json())]);
            $('#me').textContent = `You're ${me.handle}`;
            variants = cfg.variants || [];
            const pill = $('#modePill');
            pill.textContent = cfg.mode === 'live' ? 'Live models' : 'Simulated';
            pill.classList.toggle('live', cfg.mode === 'live');
            if (cfg.mode !== 'live') $('#sendSub').textContent = 'replays timings measured on H200s';
        } catch (e) {
            $('#me').textContent = 'Offline';
        }
        const saved = storeGet('qs-variant');
        selected = variants.some(v => v.key === saved) ? saved : (variants.find(v => v.key === 'INT4') || variants[0] || {}).key || null;
        buildVariants();
        connect();
    }
    boot();
})();
