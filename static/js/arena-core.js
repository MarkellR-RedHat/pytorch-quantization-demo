/*
 * Quantization Arena, simulation core (no DOM). Runs in the browser and in Node for tests.
 *
 * The course and bird physics are adapted from FlappyLearning by xviniette
 * (https://github.com/xviniette/FlappyLearning, MIT License, see THIRD_PARTY_NOTICES.md).
 * The neuroevolution half of that project is replaced by policies trained in PyTorch
 * (arena/train_policy.py). Physics must stay identical to that script;
 * tests/test_arena.py checks the two frame for frame.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.ArenaCore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    let WORLD = null;

    function configure(world) { WORLD = world; }

    function mulberry32(seed) {
        let a = seed >>> 0;
        return function () {
            a = (a + 0x6D2B79F5) >>> 0;
            let t = a;
            t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
            t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    class Course {
        constructor(seed, hardFrac) {
            this.rng = mulberry32(seed);
            this.hardFrac = hardFrac === undefined ? WORLD.HARD_FRAC : hardFrac;
            this.nextId = 0;
            this.forced = [];          // queued hard prompts from the audience or presenter
            this.pipes = [this.make(WORLD.W + 40)];
        }

        make(x) {
            const r1 = this.rng(), r2 = this.rng();
            let hard = r1 < this.hardFrac, tag = null;
            if (this.forced.length) { hard = true; tag = this.forced.shift(); }
            const gap = hard ? WORLD.GAP_HARD : WORLD.GAP_EASY;
            const lo = WORLD.MARGIN + gap / 2, hi = WORLD.GROUND - WORLD.MARGIN - gap / 2;
            const c = lo + r2 * (hi - lo);
            this.nextId += 1;
            return { id: this.nextId, x, top: c - gap / 2, bot: c + gap / 2, hard, tag };
        }

        step() {
            for (const p of this.pipes) p.x -= WORLD.SPEED;
            this.pipes = this.pipes.filter(p => p.x + WORLD.PIPE_W > -10);
            if (this.pipes[this.pipes.length - 1].x < WORLD.W - WORLD.SPACING) {
                this.pipes.push(this.make(this.pipes[this.pipes.length - 1].x + WORLD.SPACING));
            }
        }

        ahead(n, bx) { return aheadOf(this.pipes, n, bx); }
    }

    // Every function takes an optional bird x (bx). Birds fly in a staggered formation, and
    // because features are relative to the bird, each one sees the same course, shifted in time.
    function aheadOf(pipes, n, bx) {
        const x0 = (bx === undefined ? WORLD.BX : bx) - WORLD.BW / 2, out = [];
        for (const p of pipes) {
            if (p.x + WORLD.PIPE_W >= x0) { out.push(p); if (out.length === n) break; }
        }
        return out;
    }

    function birdStep(y, vy, flap) {
        vy = flap ? WORLD.FLAP : Math.min(vy + WORLD.G, WORLD.VMAX);
        return [y + vy, vy];
    }

    function isDead(y, pipes, bx) {
        const W = WORLD, x = bx === undefined ? W.BX : bx;
        if (y - W.BH / 2 < 0 || y + W.BH / 2 > W.GROUND) return true;
        for (const p of pipes) {
            if (x + W.BW / 2 > p.x && x - W.BW / 2 < p.x + W.PIPE_W) {
                if (y - W.BH / 2 < p.top || y + W.BH / 2 > p.bot) return true;
            }
        }
        return false;
    }

    function features(y, vy, ahead, bx) {
        const x = bx === undefined ? WORLD.BX : bx;
        const p0 = ahead[0], p1 = ahead.length > 1 ? ahead[1] : ahead[0];
        return [
            p0.bot - y,
            (y - p0.top) / 100,
            vy / WORLD.VMAX,
            (p0.x - x) / 300,
            (p1.bot - y) / 100,
            (y - p1.top) / 100,
        ];
    }

    // Same op order as numpy's (W @ h + b) for a single vector: row-wise dot products.
    function forward(layers, x) {
        let h = x;
        for (let l = 0; l < layers.length; l++) {
            const Wm = layers[l].W, b = layers[l].b, out = new Array(Wm.length);
            for (let i = 0; i < Wm.length; i++) {
                const row = Wm[i];
                let s = 0;
                for (let j = 0; j < row.length; j++) s += row[j] * h[j];
                s += b[i];
                out[i] = (l < layers.length - 1 && s < 0) ? 0 : s;
            }
            h = out;
        }
        return h[0];
    }

    function act(layers, y, vy, ahead, bx) { return forward(layers, features(y, vy, ahead, bx)) > 0 ? 1 : 0; }

    /*
     * Greedy speculative decoding over actions. The draft rolls the world forward K frames
     * on its own proposals, the target scores all K+1 states in one (batched) pass, and we
     * keep the agreeing prefix plus the target's own action at the first disagreement, or a
     * bonus action when everything was accepted. The committed actions are exactly what the
     * target alone would have chosen, so the flight path matches the BF16 bird to the pixel.
     */
    function speculate(target, draft, y, vy, pipes, K, bx) {
        const ghost = pipes.map(p => ({ x: p.x, top: p.top, bot: p.bot }));
        const states = [], proposals = [];
        let ys = y, vs = vy;
        for (let k = 0; k < K; k++) {
            const ah = aheadOf(ghost, 2, bx);
            states.push([ys, vs, ah.map(p => ({ ...p }))]);
            const a = act(draft, ys, vs, ah, bx);
            [ys, vs] = birdStep(ys, vs, a);
            proposals.push({ a, y: ys });
            for (const p of ghost) p.x -= WORLD.SPEED;
        }
        states.push([ys, vs, aheadOf(ghost, 2, bx)]);
        const verdict = states.map(s => act(target, s[0], s[1], s[2], bx));
        let n = 0;
        while (n < K && proposals[n].a === verdict[n]) n++;
        const commit = proposals.slice(0, n).map(p => p.a);
        commit.push(verdict[n]);
        return { commit, accepted: n, proposals: proposals.map((p, i) => ({ y: p.y, ok: i < n })) };
    }

    return { configure, mulberry32, Course, aheadOf, birdStep, isDead, features, forward, act, speculate };
});
