/*
 * Quantization Arena, game loop and renderer.
 *
 * Game loop structure adapted from FlappyLearning by xviniette (MIT, see THIRD_PARTY_NOTICES.md).
 * Every AI bird runs the same PyTorch-trained policy (static/arena/policy.json); only the
 * precision of its weights differs. Nothing about a bird's flight is scripted.
 */
(function () {
    'use strict';
    const Core = window.ArenaCore;
    const STEP_MS = 1000 / 60;
    const TRAIL = 14;           // frames of trail, which reaches exactly to the bird behind
    const RESPAWN_DELAY = 36;   // frames a crashed bird is gone
    const GHOST_FRAMES = 70;    // frames of invulnerability after respawn

    const ROSTER = [
        { key: 'SPEC', tag: 'SPEC', name: 'Spec Decode', color: '--v-spec', kind: 'spec', bx: 344 },
        { key: 'BF16', tag: 'BF16', name: 'BF16', color: '--v-bf16', kind: 'net', bx: 288 },
        { key: 'FP8', tag: 'FP8', name: 'FP8', color: '--v-fp8', kind: 'net', bx: 232 },
        { key: 'INT4_AWQ', tag: 'AWQ', name: 'INT4 AWQ', color: '--v-int4', kind: 'net', bx: 176 },
        { key: 'INT4_RTN', tag: 'RTN', name: 'INT4 RTN', color: '--v-int4', kind: 'net', bx: 120, hollow: true, dashed: true },
    ];
    const HUMAN = { key: 'YOU', tag: 'YOU', name: 'You', color: '--v-you', kind: 'human', bx: 452 };

    function cssVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    class Arena {
        constructor(canvas, policy, opts = {}) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.policy = policy;
            this.world = policy.world;
            Core.configure(this.world);
            this.onChange = opts.onChange || (() => {});
            this.onEvent = opts.onEvent || (() => {});
            this.K = 4;
            this.paused = false;
            this.acc = 0;
            this.last = performance.now();
            this.frame = 0;
            this.fx = [];
            this.floaters = [];
            this.hidden = new Set();
            this.readColors();
            this.reset(opts.seed || 20261020);
            requestAnimationFrame(t => this.loop(t));
        }

        readColors() {
            const c = {};
            for (const n of ['--sky-top', '--sky-bottom', '--grid', '--pipe', '--pipe-edge', '--pipe-hard',
                '--pipe-hard-edge', '--ground', '--ground-tick', '--ink', '--ink-2', '--ink-3', '--surface',
                '--line-strong', '--brand', '--v-bf16', '--v-fp8', '--v-int4', '--v-spec', '--v-you']) c[n] = cssVar(n);
            this.colors = c;
        }

        reset(seed) {
            this.seed = seed;
            this.course = new Core.Course(seed);
            this.frame = 0;
            this.fx = [];
            this.floaters = [];
            const human = this.birds && this.birds.find(b => b.kind === 'human');
            this.birds = ROSTER.map(r => this.makeBird(r));
            if (human) this.birds.unshift(this.makeBird(HUMAN));
            this.onChange(this.snapshot());
        }

        makeBird(r) {
            const b = {
                ...r, y: 250, vy: 0, down: 0, ghost: 0, flapT: 0, trail: [],
                scored: new Set(),
                stats: { easy: [0, 0], hard: [0, 0], crashes: 0 },
            };
            if (r.kind === 'net') b.layers = this.policy.variants[r.key].layers;
            if (r.kind === 'spec') {
                b.layers = this.policy.variants.BF16.layers;
                b.draft = this.policy.draft.layers;
                b.queue = [];
                b.spec = { passes: 0, proposed: 0, accepted: 0, frames: 0, strips: [], maxDiff: 0 };
                // An invisible BF16 bird flying the exact same spot, to measure the path difference.
                b.shadow = { y: 250, vy: 0 };
            }
            return b;
        }

        throwHard(label, from) {
            this.course.forced.push({ label, from: from || null });
        }

        flapHuman() {
            let h = this.birds.find(b => b.kind === 'human');
            if (!h) { h = this.makeBird(HUMAN); this.birds.unshift(h); }
            if (!h.down) { h.vy = this.world.FLAP; h.flapT = 1; h.humanFlap = true; }
        }

        removeHuman() { this.birds = this.birds.filter(b => b.kind !== 'human'); }

        toggle(key) {
            if (this.hidden.has(key)) this.hidden.delete(key); else this.hidden.add(key);
        }

        // simulation

        loop(t) {
            const dt = Math.min(250, t - this.last);
            this.last = t;
            if (!this.paused) {
                this.acc += dt;
                let n = 0;
                while (this.acc >= STEP_MS && n < 8) { this.tick(); this.acc -= STEP_MS; n++; }
                if (n === 8) this.acc = 0;
            }
            this.render();
            requestAnimationFrame(tt => this.loop(tt));
        }

        tick() {
            const W = this.world;
            this.frame++;
            for (const b of this.birds) this.tickBird(b);
            this.course.step();
            for (const b of this.birds) this.scoreBird(b);
            for (const p of this.fx) { p.x += p.vx; p.y += p.vy; p.vy += 0.18; p.life -= 0.022; }
            this.fx = this.fx.filter(p => p.life > 0);
            for (const f of this.floaters) { f.y -= 0.45; f.life -= 0.012; f.x -= W.SPEED * 0.5; }
            this.floaters = this.floaters.filter(f => f.life > 0);
            if (this.frame % 30 === 0) this.onChange(this.snapshot());
        }

        tickBird(b) {
            const W = this.world, pipes = this.course.pipes;
            if (b.down > 0) {
                b.down--;
                if (b.down === 0) this.respawn(b);
                return;
            }
            let a;
            if (b.kind === 'human') {
                a = b.humanFlap ? 1 : 0;
                b.humanFlap = false;
                if (a) { [b.y, b.vy] = [b.y + W.FLAP, W.FLAP]; } else { [b.y, b.vy] = Core.birdStep(b.y, b.vy, 0); }
            } else {
                if (b.kind === 'spec') {
                    if (!b.queue.length) {
                        const r = Core.speculate(b.layers, b.draft, b.y, b.vy, pipes, this.K, b.bx);
                        b.queue = r.commit.slice();
                        b.spec.passes++;
                        b.spec.proposed += this.K;
                        b.spec.accepted += r.accepted;
                        b.spec.strips.push({ accepted: r.accepted, k: this.K });
                        if (b.spec.strips.length > 14) b.spec.strips.shift();
                    }
                    a = b.queue.shift();
                    b.spec.frames++;
                    const s = b.shadow;
                    const sa = Core.act(b.layers, s.y, s.vy, Core.aheadOf(pipes, 2, b.bx), b.bx);
                    [s.y, s.vy] = Core.birdStep(s.y, s.vy, sa);
                } else {
                    a = Core.act(b.layers, b.y, b.vy, Core.aheadOf(pipes, 2, b.bx), b.bx);
                }
                [b.y, b.vy] = Core.birdStep(b.y, b.vy, a);
                if (b.kind === 'spec') b.spec.maxDiff = Math.max(b.spec.maxDiff, Math.abs(b.y - b.shadow.y));
            }
            if (a) b.flapT = 1;
            b.flapT = Math.max(0, b.flapT - 0.12);
            b.trail.unshift(b.y);
            if (b.trail.length > TRAIL) b.trail.pop();
            if (b.ghost > 0) { b.ghost--; return; }
            // Check against the pipes as they will be after this frame's scroll, like the Python eval.
            const next = pipes.map(p => ({ x: p.x - W.SPEED, top: p.top, bot: p.bot }));
            if (Core.isDead(b.y, next, b.bx)) this.crash(b);
        }

        scoreBird(b) {
            const W = this.world;
            for (const p of this.course.pipes) {
                if (p.x + W.PIPE_W < b.bx - W.BW / 2 && !b.scored.has(p.id)) {
                    b.scored.add(p.id);
                    const s = p.hard ? b.stats.hard : b.stats.easy;
                    s[0]++; s[1]++;
                }
            }
        }

        crash(b) {
            const W = this.world;
            const at = this.course.pipes.filter(p => p.x - W.SPEED <= b.bx + W.BW / 2 && p.x - W.SPEED + W.PIPE_W >= b.bx - W.BW / 2);
            let label = 'Hit the ground';
            for (const p of at) {
                if (!b.scored.has(p.id)) {
                    b.scored.add(p.id);
                    (p.hard ? b.stats.hard : b.stats.easy)[1]++;
                }
                label = p.hard ? (p.tag ? p.tag.label : 'Hard prompt') : 'Easy prompt';
            }
            b.stats.crashes++;
            const col = this.colors[b.color];
            for (let i = 0; i < 22; i++) {
                const ang = Math.random() * Math.PI * 2, sp = 1.5 + Math.random() * 4.5;
                this.fx.push({ x: b.bx, y: b.y, vx: Math.cos(ang) * sp - 1.5, vy: Math.sin(ang) * sp - 2, life: 1, r: 2 + Math.random() * 3.5, c: col });
            }
            this.floaters.push({ x: b.bx, y: b.y - 34, text: `${b.tag} crashed · ${label}`, life: 1 });
            b.down = RESPAWN_DELAY;
            b.trail = [];
            this.onEvent({ type: 'crash', bird: b.key, label });
            this.onChange(this.snapshot());
        }

        respawn(b) {
            const W = this.world;
            const ahead = Core.aheadOf(this.course.pipes, 2, b.bx);
            for (const p of ahead) if (p.x <= b.bx + W.BW / 2 + 8) b.scored.add(p.id);
            const nx = ahead.find(p => !b.scored.has(p.id)) || ahead[0];
            b.y = nx ? (nx.top + nx.bot) / 2 : W.GROUND / 2;
            b.vy = 0;
            b.ghost = GHOST_FRAMES;
            if (b.kind === 'spec') { b.queue = []; b.shadow = { y: b.y, vy: 0 }; }
        }

        snapshot() {
            const birds = {};
            for (const b of this.birds) {
                if (b.kind === 'human') continue;
                birds[b.key] = { name: b.name, easy: b.stats.easy.slice(), hard: b.stats.hard.slice(), crashes: b.stats.crashes };
            }
            const sb = this.birds.find(b => b.kind === 'spec');
            const spec = sb && sb.spec.passes ? {
                k: this.K,
                acceptance: sb.spec.accepted / sb.spec.proposed,
                frames_per_pass: sb.spec.frames / sb.spec.passes,
                max_diff_px: sb.spec.maxDiff,
                strips: sb.spec.strips.slice(),
            } : null;
            const gaps = this.course.nextId;
            return { frame: this.frame, gaps, birds, spec };
        }

        // rendering

        resize(pxW, pxH) {
            if (this.canvas.width !== pxW || this.canvas.height !== pxH) {
                this.canvas.width = pxW;
                this.canvas.height = pxH;
            }
        }

        render() {
            const ctx = this.ctx, W = this.world, C = this.colors;
            const k = this.canvas.width / W.W;
            ctx.setTransform(k, 0, 0, k, 0, 0);

            const sky = ctx.createLinearGradient(0, 0, 0, W.GROUND);
            sky.addColorStop(0, C['--sky-top']);
            sky.addColorStop(1, C['--sky-bottom']);
            ctx.fillStyle = sky;
            ctx.fillRect(0, 0, W.W, W.H);

            // Graph-paper grid that scrolls with the course
            const off = (this.frame * W.SPEED) % 30;
            ctx.strokeStyle = C['--grid'];
            ctx.lineWidth = 1 / k;
            ctx.beginPath();
            for (let x = -off; x < W.W; x += 30) { ctx.moveTo(x, 0); ctx.lineTo(x, W.GROUND); }
            for (let y = 20; y < W.GROUND; y += 30) { ctx.moveTo(0, y); ctx.lineTo(W.W, y); }
            ctx.stroke();

            for (const p of this.course.pipes) this.drawPipe(p);
            this.drawGround();
            for (const p of this.course.pipes) this.drawPipeLabel(p);
            for (const b of this.birds) if (!this.hidden.has(b.key)) this.drawTrail(b);
            for (let i = this.birds.length - 1; i >= 0; i--) {
                const b = this.birds[i];
                if (!this.hidden.has(b.key) && !b.down) this.drawBird(b);
            }
            for (const b of this.birds) if (!this.hidden.has(b.key) && !b.down) this.drawTag(b);
            this.drawSpecStrip();

            for (const p of this.fx) {
                ctx.globalAlpha = Math.max(0, p.life);
                ctx.fillStyle = p.c;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r * (0.4 + 0.6 * p.life), 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.globalAlpha = 1;
            for (const f of this.floaters) this.drawFloater(f);

            if (this.paused) {
                ctx.fillStyle = 'rgba(21,21,21,0.28)';
                ctx.fillRect(0, 0, W.W, W.H);
                this.chip(W.W / 2, W.H / 2 - 20, 'Paused · press P to resume', C['--surface'], C['--ink'], 18, true);
            }
        }

        roundRect(x, y, w, h, r) {
            const ctx = this.ctx;
            r = Math.min(r, w / 2, h / 2);
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.arcTo(x + w, y, x + w, y + h, r);
            ctx.arcTo(x + w, y + h, x, y + h, r);
            ctx.arcTo(x, y + h, x, y, r);
            ctx.arcTo(x, y, x + w, y, r);
            ctx.closePath();
        }

        drawPipe(p) {
            const ctx = this.ctx, W = this.world, C = this.colors;
            const body = p.hard ? C['--pipe-hard'] : C['--pipe'];
            const edge = p.hard ? C['--pipe-hard-edge'] : C['--pipe-edge'];
            const lip = 18, over = 5;
            ctx.fillStyle = body;
            this.roundRect(p.x, -10, W.PIPE_W, p.top + 10 - lip + 2, 4); ctx.fill();
            this.roundRect(p.x, p.bot + lip - 2, W.PIPE_W, W.GROUND - p.bot - lip + 2, 4); ctx.fill();
            ctx.fillStyle = edge;
            this.roundRect(p.x - over, p.top - lip, W.PIPE_W + over * 2, lip, 5); ctx.fill();
            this.roundRect(p.x - over, p.bot, W.PIPE_W + over * 2, lip, 5); ctx.fill();
            // soft highlight for depth
            ctx.fillStyle = 'rgba(255,255,255,0.10)';
            ctx.fillRect(p.x + 10, -10, 10, p.top - lip + 10);
            ctx.fillRect(p.x + 10, p.bot + lip, 10, W.GROUND - p.bot - lip);
        }

        drawPipeLabel(p) {
            if (!p.hard) return;
            const W = this.world, C = this.colors;
            const label = p.tag ? p.tag.label : 'Hard prompt';
            const who = p.tag && p.tag.from ? ` · ${p.tag.from}` : '';
            this.chip(p.x + W.PIPE_W / 2, W.GROUND + (W.H - W.GROUND) / 2 - 2, label + who, C['--pipe-hard'], '#ffffff', 12.5, true);
        }

        drawGround() {
            const ctx = this.ctx, W = this.world, C = this.colors;
            ctx.fillStyle = C['--ground'];
            ctx.fillRect(0, W.GROUND, W.W, W.H - W.GROUND);
            ctx.fillStyle = C['--line-strong'];
            ctx.fillRect(0, W.GROUND, W.W, 1.5);
            ctx.fillStyle = C['--ground-tick'];
            const off = (this.frame * W.SPEED) % 24;
            for (let x = -off; x < W.W; x += 24) ctx.fillRect(x, W.H - 8, 12, 3);
        }

        drawTrail(b) {
            if (b.down || b.trail.length < 2) return;
            const ctx = this.ctx, W = this.world;
            ctx.save();
            ctx.strokeStyle = this.colors[b.color];
            ctx.lineWidth = 2.5;
            ctx.lineCap = 'round';
            if (b.dashed) ctx.setLineDash([5, 6]);
            ctx.globalAlpha = b.ghost ? 0.2 : 0.45;
            ctx.beginPath();
            for (let i = 0; i < b.trail.length; i++) {
                const x = b.bx - i * W.SPEED;
                if (i === 0) ctx.moveTo(x, b.trail[i]); else ctx.lineTo(x, b.trail[i]);
            }
            ctx.stroke();
            ctx.restore();
        }

        drawBird(b) {
            const ctx = this.ctx, W = this.world, C = this.colors;
            const col = C[b.color];
            const rx = W.BW / 2, ry = W.BH / 2;
            ctx.save();
            ctx.translate(b.bx, b.y);
            ctx.rotate(Math.max(-0.45, Math.min(0.85, b.vy * 0.07)));
            ctx.globalAlpha = b.ghost ? (0.35 + 0.25 * Math.sin(this.frame * 0.4)) : 1;
            // body
            ctx.beginPath();
            ctx.ellipse(0, 0, rx - 1.5, ry - 1.5, 0, 0, Math.PI * 2);
            ctx.fillStyle = b.hollow ? C['--surface'] : col;
            ctx.fill();
            ctx.lineWidth = 3;
            ctx.strokeStyle = col;
            ctx.stroke();
            // wing
            const lift = b.flapT;
            ctx.beginPath();
            ctx.ellipse(-4, 2 - lift * 5, 8, 4.5, -0.35 - lift * 0.6, 0, Math.PI * 2);
            ctx.fillStyle = b.hollow ? col : 'rgba(255,255,255,0.55)';
            ctx.fill();
            // eye
            ctx.beginPath();
            ctx.arc(8, -4, 4.2, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(9.3, -4, 2, 0, Math.PI * 2);
            ctx.fillStyle = '#151515';
            ctx.fill();
            // beak
            ctx.beginPath();
            ctx.moveTo(rx - 3, 1);
            ctx.lineTo(rx + 6, 3.5);
            ctx.lineTo(rx - 3, 6.5);
            ctx.closePath();
            ctx.fillStyle = C['--brand'];
            ctx.fill();
            ctx.restore();
        }

        drawTag(b) {
            const C = this.colors;
            const col = C[b.color];
            const fg = b.kind === 'human' ? C['--surface'] : '#ffffff';
            this.chip(b.bx, b.y - 32, b.tag, b.hollow ? C['--surface'] : col, b.hollow ? col : fg, 11.5, true, b.hollow ? col : null, b.ghost ? 0.5 : 1);
        }

        drawSpecStrip() {
            const sb = this.birds.find(b => b.kind === 'spec');
            if (!sb || sb.down || this.hidden.has('SPEC') || !sb.spec.strips.length) return;
            const ctx = this.ctx, C = this.colors, last = sb.spec.strips[sb.spec.strips.length - 1];
            const x0 = sb.bx + 26, y0 = sb.y - 4, s = 7, gap = 2.5;
            for (let i = 0; i <= last.k; i++) {
                const x = x0 + i * (s + gap);
                this.roundRect(x, y0, s, s, 1.5);
                if (i < last.accepted) { ctx.fillStyle = C['--v-spec']; ctx.fill(); }
                else if (i === last.accepted) { ctx.fillStyle = C['--surface']; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = C['--v-spec']; ctx.stroke(); }
                else { ctx.fillStyle = 'rgba(128,128,128,0.25)'; ctx.fill(); }
            }
        }

        drawFloater(f) {
            const C = this.colors;
            this.chip(f.x, f.y, f.text, C['--surface'], C['--brand'], 12, true, C['--brand'], Math.min(1, f.life * 1.6));
        }

        chip(cx, cy, text, bg, fg, size, bold, border, alpha = 1) {
            const ctx = this.ctx;
            ctx.save();
            ctx.globalAlpha = alpha;
            ctx.font = `${bold ? 700 : 500} ${size}px 'Red Hat Mono', ui-monospace, monospace`;
            const w = ctx.measureText(text).width + size * 1.1, h = size * 1.75;
            const x = Math.max(4, Math.min(this.world.W - w - 4, cx - w / 2)), y = cy - h / 2;
            this.roundRect(x, y, w, h, h / 2);
            ctx.fillStyle = bg;
            ctx.fill();
            if (border) { ctx.lineWidth = 1.5; ctx.strokeStyle = border; ctx.stroke(); }
            ctx.fillStyle = fg;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, x + w / 2, cy + size * 0.06);
            ctx.restore();
        }
    }

    window.QuantArena = { Arena, ROSTER };
})();
