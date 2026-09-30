#!/usr/bin/env python3
"""Screenshot the presenter view for every Ask preset, then once more after R (recorded mode).

Run it while the app is up (python -m uvicorn app.main:app --port 8000) and the port-forwards are live.

    pip install playwright && playwright install chromium
    python3 screenshot_presenter.py                       # http://localhost:8000, key from .env
    python3 screenshot_presenter.py --url http://localhost:8000 --key <PRESENTER_KEY> --out results-2/live

Writes <out>/<phase>-<preset>.png for each preset, <out>/<phase>-badge.txt with the corner badge text,
<out>/<phase>-recorded-mode.png after pressing R, and <out>/errors.txt for anything that failed.
Nothing gets typed in by hand: every screenshot is the real page.
"""
import argparse
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright


def read_env_key(path=".env"):
    try:
        for line in open(path):
            m = re.match(r"\s*PRESENTER_KEY\s*=\s*(.*)\s*$", line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default=read_env_key())
    ap.add_argument("--out", default="results-2/live")
    ap.add_argument("--phase", default="phaseA", help="prefix for the file names, for example phaseA or phaseB")
    ap.add_argument("--timeout", type=int, default=180, help="seconds to wait for all columns per preset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    errors = []

    def note(msg):
        print(msg)
        errors.append(msg)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.on("pageerror", lambda e: note(f"page error: {e}"))
        page.on("console", lambda m: note(f"console {m.type}: {m.text}") if m.type == "error" else None)

        url = a.url.rstrip("/") + "/presenter" + (f"?key={a.key}" if a.key else "")
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("#presets button", timeout=30000)
        page.wait_for_timeout(1500)
        badge = page.inner_text("#modeBadge").strip()
        open(os.path.join(a.out, f"{a.phase}-badge.txt"), "w").write(badge + "\n")
        print(f"badge: {badge}")
        if "Live" not in badge:
            note(f"badge does not say Live models: '{badge}'")

        presets = page.eval_on_selector_all("#presets button", "bs => bs.map(b => [b.dataset.preset, b.textContent])")
        print(f"{len(presets)} presets: {[k for k, _ in presets]}")
        if len(presets) != 8:
            note(f"expected 8 presets, found {len(presets)}")

        for key, label in presets:
            page.click(f"#presets button[data-preset='{key}']")
            t0 = time.time()
            # the click resets every column's token count to '…'; wait for that before waiting for the answers
            page.wait_for_function("() => [...document.querySelectorAll('#askGrid .acol .s-len')].every(e => e.textContent.trim() === '…')", timeout=10000)
            # a column is finished when its token count is filled in (the done event) or it shows an error
            done = False
            while time.time() - t0 < a.timeout:
                states = page.eval_on_selector_all(
                    "#askGrid .acol",
                    "cols => cols.map(c => ({key: c.dataset.key, place: c.querySelector('.place')?.textContent || '', "
                    "len: (c.querySelector('.s-len')?.textContent || '…').trim(), "
                    "err: c.querySelector('.answer')?.classList.contains('err') || false, "
                    "rec: !!c.querySelector('.fb-divider')}))",
                )
                if states and all(s["len"] != "…" or s["err"] for s in states):
                    done = True
                    break
                page.wait_for_timeout(500)
            page.wait_for_timeout(800)
            path = os.path.join(a.out, f"{a.phase}-{key}.png")
            page.screenshot(path=path, full_page=False)
            summary = ", ".join(f"{s['key']}={'error' if s['err'] else ('recorded' if s['rec'] else (s['place'] or (s['len'] + ' tokens')) if s['len'] != '…' else 'unfinished')}" for s in states)
            print(f"{label}: {summary} ({time.time() - t0:.0f}s) -> {path}")
            if not done:
                note(f"{label}: not every column finished within {a.timeout}s ({summary})")
            for s in states:
                if s["err"]:
                    note(f"{label}: {s['key']} column errored")
                if s["rec"]:
                    note(f"{label}: {s['key']} column fell back to its recording")

        # R flips every column to recorded mode; one screenshot proves the badge and label change
        page.keyboard.press("KeyR")
        page.wait_for_timeout(1500)
        badge_r = page.inner_text("#modeBadge").strip()
        print(f"badge after R: {badge_r}")
        page.click("#presets button >> nth=0")
        page.wait_for_timeout(12000)
        page.screenshot(path=os.path.join(a.out, f"{a.phase}-recorded-mode.png"))
        page.keyboard.press("KeyR")
        page.wait_for_timeout(1000)
        print(f"badge after second R: {page.inner_text('#modeBadge').strip()}")
        browser.close()

    with open(os.path.join(a.out, "errors.txt"), "a") as f:
        f.write(f"# {a.phase} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n".join(errors) + ("\n" if errors else "no errors\n"))
    print(f"{len(errors)} problem(s) noted in {a.out}/errors.txt")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
