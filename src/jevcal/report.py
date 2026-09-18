"""Self-contained HTML report: no external assets, light and dark themes, inline SVG charts."""

from __future__ import annotations

import html
from pathlib import Path

W, H = 480, 300
PAD_L, PAD_R, PAD_T, PAD_B = 44, 16, 18, 40

CSS = """
:root { color-scheme: light;
  --surface-0:#f4f3f0; --surface-1:#fcfcfb; --border:#e3e1da; --grid:#ebe9e3;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#85837c;
  --series-1:#2a78d6; --series-2:#eb6834;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --surface-0:#111110; --surface-1:#1a1a19; --border:#33332f; --grid:#2a2a27;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c83;
  --series-1:#3987e5; --series-2:#d95926; } }
:root[data-theme="dark"] { color-scheme: dark;
  --surface-0:#111110; --surface-1:#1a1a19; --border:#33332f; --grid:#2a2a27;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c83;
  --series-1:#3987e5; --series-2:#d95926; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface-0); color:var(--text-primary);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:19px; margin:0; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
h3 { font-size:13px; margin:0 0 8px; color:var(--text-secondary); font-weight:600; }
.meta { color:var(--text-secondary); margin:0 0 24px; font-size:14px; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:20px; margin:0 0 20px; }
.qhead { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:6px; }
.qtext { color:var(--text-secondary); margin:0 0 16px; font-size:14px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:0 0 18px; }
.tile { border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
.tile .k { font-size:12px; color:var(--text-secondary); }
.tile .v { font-size:24px; font-weight:650; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }
.tile .s { font-size:12px; color:var(--text-muted); }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; }
svg { width:100%; height:auto; display:block; }
svg text { font:11px ui-sans-serif,system-ui,sans-serif; fill:var(--text-secondary); }
svg .muted { fill:var(--text-muted); }
svg .strong { fill:var(--text-primary); font-weight:600; }
.legend { display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--text-secondary); margin:0 0 4px; }
.legend i { display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }
.badge { display:inline-flex; align-items:center; gap:6px; font-size:12px; padding:3px 9px; border-radius:999px;
  border:1px solid var(--border); color:var(--text-primary); }
.badge b { display:inline-grid; place-items:center; width:15px; height:15px; border-radius:50%; color:#fff; font-size:10px; }
.note { border-left:3px solid var(--warning); padding:6px 12px; margin:12px 0 0; font-size:14px; color:var(--text-secondary); }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th,td { text-align:right; padding:5px 10px; border-bottom:1px solid var(--border); }
th:first-child,td:first-child { text-align:left; }
th { color:var(--text-secondary); font-weight:600; }
details { margin-top:14px; } summary { cursor:pointer; color:var(--text-secondary); font-size:13px; }
.scroll { overflow-x:auto; }
#tip { position:fixed; pointer-events:none; background:var(--text-primary); color:var(--surface-1); padding:6px 9px;
  border-radius:6px; font-size:12px; line-height:1.35; opacity:0; transition:opacity .08s; white-space:pre; z-index:9; }
.hit { fill:transparent; } .hit:hover + .mark, .mark:hover { filter:brightness(1.12); }
footer { color:var(--text-muted); font-size:12px; margin-top:24px; }
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
 if(!t){tip.style.opacity=0;return;}
 tip.textContent=t.dataset.tip;tip.style.opacity=1;
 const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=Math.max(8,e.clientY-tip.offsetHeight-10);
 tip.style.left=x+'px';tip.style.top=y+'px';});
"""

STATUS = {
    "ok": ("good", "&#10003;", "Threshold holds on held-out data"),
    "holdout_miss": ("warning", "!", "Threshold does not generalize"),
    "no_threshold": ("critical", "&#10005;", "No safe threshold: escalate everything"),
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pct(value, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _sx(x: float, lo: float) -> float:
    span = (1.0 - lo) or 1.0
    return PAD_L + (x - lo) / span * (W - PAD_L - PAD_R)


def _sy(y: float) -> float:
    return PAD_T + (1.0 - y) * (H - PAD_T - PAD_B)


def _frame(lo: float, x_label: str, y_label: str) -> str:
    parts = []
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = _sy(tick)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        parts.append(f'<text class="muted" x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">{int(tick * 100)}%</text>')
    for i in range(5):
        value = lo + (1.0 - lo) * i / 4
        parts.append(f'<text class="muted" x="{_sx(value, lo):.1f}" y="{H - PAD_B + 16}" text-anchor="middle">{value:.2f}</text>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.1f}" y="{H - 6}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text x="12" y="{(PAD_T + H - PAD_B) / 2:.1f}" text-anchor="middle" transform="rotate(-90 12 {(PAD_T + H - PAD_B) / 2:.1f})">{esc(y_label)}</text>')
    return "".join(parts)


def _bar(x0: float, x1: float, y_top: float, y_base: float) -> str:
    """Bar anchored to the baseline with a 4px rounded data-end."""
    r = min(4.0, (x1 - x0) / 2, max(0.0, y_base - y_top))
    return (f"M{x0:.1f},{y_base:.1f} L{x0:.1f},{y_top + r:.1f} Q{x0:.1f},{y_top:.1f} {x0 + r:.1f},{y_top:.1f} "
            f"L{x1 - r:.1f},{y_top:.1f} Q{x1:.1f},{y_top:.1f} {x1:.1f},{y_top + r:.1f} L{x1:.1f},{y_base:.1f} Z")


def reliability_svg(bins: list[dict]) -> str:
    if not bins:
        return "<p class='qtext'>No labeled rows.</p>"
    lo = min(b["lo"] for b in bins)
    lo = 0.0 if lo < 0.05 else lo
    body = [_frame(lo, "Stated confidence", "Observed accuracy")]
    for b in bins:
        x0, x1 = _sx(b["lo"], lo) + 1, _sx(b["hi"], lo) - 1  # 2px surface gap between fills
        tip = (f"Confidence {b['lo']:.2f} to {b['hi']:.2f}\nstated {pct(b['confidence'])}  actual {pct(b['accuracy'])}\n"
               f"{b['n']} decisions")
        body.append(f'<rect class="hit" x="{x0:.1f}" y="{PAD_T}" width="{x1 - x0:.1f}" height="{H - PAD_T - PAD_B}" data-tip="{esc(tip)}"/>')
        body.append(f'<path class="mark" d="{_bar(x0, x1, _sy(b["accuracy"]), _sy(0))}" fill="var(--series-1)" data-tip="{esc(tip)}"/>')
    # drawn over the bars: a bar that stops short of this line is overconfident
    body.append(f'<line x1="{_sx(lo, lo):.1f}" y1="{_sy(lo):.1f}" x2="{_sx(1, lo):.1f}" y2="{_sy(1):.1f}" '
                f'stroke="var(--text-primary)" stroke-width="1.5" stroke-dasharray="4 4" pointer-events="none"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Reliability diagram">{"".join(body)}</svg>'


def coverage_svg(points: list[dict], threshold: float | None, target: float) -> str:
    if len(points) < 2:
        return "<p class='qtext'>Not enough distinct confidence values to draw a curve.</p>"
    lo = min(p["threshold"] for p in points)
    lo = max(0.0, min(lo, 0.95))
    body = [_frame(lo, "Confidence threshold", "Share of decisions")]
    ty = _sy(target)
    body.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="var(--text-muted)" stroke-width="1.5" stroke-dasharray="4 4"/>')
    body.append(f'<text class="muted" x="{PAD_L + 6}" y="{ty + 14:.1f}">target {pct(target, 0)}</text>')

    def path(key: str) -> str:
        return " ".join(f"{'M' if i == 0 else 'L'}{_sx(p['threshold'], lo):.1f},{_sy(p[key]):.1f}" for i, p in enumerate(points))

    body.append(f'<path d="{path("coverage")}" fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round"/>')
    body.append(f'<path d="{path("accuracy")}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>')

    # direct labels at the high-threshold end, where the two lines are furthest apart
    last = points[-1]
    rx = _sx(last["threshold"], lo) - 6
    body.append(f'<text x="{rx:.1f}" y="{min(H - PAD_B - 6, _sy(last["accuracy"]) + 34):.1f}" text-anchor="end">accepted accuracy</text>')
    body.append(f'<text x="{rx:.1f}" y="{max(PAD_T + 40, _sy(last["coverage"]) - 8):.1f}" text-anchor="end">handled by fast model</text>')

    if threshold is not None:
        x = _sx(threshold, lo)
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H - PAD_B}" stroke="var(--text-primary)" stroke-width="1.5"/>')
        anchor, dx = ("end", -6) if x > W * 0.55 else ("start", 6)
        body.append(f'<text class="strong" x="{x + dx:.1f}" y="{PAD_T + 10}" text-anchor="{anchor}">threshold {threshold:.3f}</text>')
        at = min(points, key=lambda p: abs(p["threshold"] - threshold))
        for key, color in (("coverage", "var(--series-2)"), ("accuracy", "var(--series-1)")):
            body.append(f'<circle cx="{x:.1f}" cy="{_sy(at[key]):.1f}" r="5" fill="{color}" stroke="var(--surface-1)" stroke-width="2"/>')

    step = max(1, len(points) // 60)
    sampled = points[::step]
    for i, p in enumerate(sampled):
        left = _sx(p["threshold"], lo) if i == 0 else (_sx(sampled[i - 1]["threshold"], lo) + _sx(p["threshold"], lo)) / 2
        right = _sx(p["threshold"], lo) if i == len(sampled) - 1 else (_sx(sampled[i + 1]["threshold"], lo) + _sx(p["threshold"], lo)) / 2
        tip = (f"Threshold {p['threshold']:.3f}\naccepted accuracy {pct(p['accuracy'])}\n"
               f"handled by fast model {pct(p['coverage'])}  ({p['n']} rows)")
        body.append(f'<rect class="hit" x="{left:.1f}" y="{PAD_T}" width="{max(1.0, right - left):.1f}" height="{H - PAD_T - PAD_B}" data-tip="{esc(tip)}"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Accuracy and coverage by threshold">{"".join(body)}</svg>'


def tile(label: str, value: str, sub: str = "") -> str:
    return f'<div class="tile"><div class="k">{esc(label)}</div><div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'


def question_section(qid: str, instructions: str, result: dict, states: dict[str, str]) -> str:
    tone, icon, label = STATUS[result["status"]]
    held, cal = result["holdout"], result["calibration"]
    over = cal["overconfidence"]
    lean = "n/a" if over is None else ("overconfident" if over > 0.01 else "underconfident" if over < -0.01 else "well calibrated")
    tiles = "".join([
        tile("Threshold", num(result["threshold"]), f"on {result['measure']}"),
        tile("Handled by fast model", pct(held["coverage"]), f"held-out, n={held['n']}"),
        tile("Accepted accuracy", pct(held["accepted_accuracy"]), f"held-out, target {pct(result['target'], 0)}"),
        tile("Accuracy, no threshold", pct(result["accuracy_all"]), f"all {result['n']} rows"),
        tile("Calibration error (ECE)", pct(cal["ece"]), lean),
    ])
    answers = "".join(
        f"<tr><td>{esc(a)}</td><td>{s['n_accepted']}</td><td>{pct(s['accuracy'])}</td></tr>"
        for a, s in sorted(result["by_answer"].items())
    )
    step = max(1, len(result["sweep"]) // 25)
    sweep_rows = "".join(
        f"<tr><td>{p['threshold']:.3f}</td><td>{pct(p['coverage'])}</td><td>{pct(p['accuracy'])}</td><td>{pct(p['wilson_lb'])}</td><td>{p['n']}</td></tr>"
        for p in result["sweep"][::step]
    )
    auroc = " &middot; ".join(f"{esc(m)} {num(v)}" for m, v in cal["auroc"].items())
    notes = "".join(f'<p class="note">{esc(w)}</p>' for w in result["warnings"])
    misses = "".join(
        f"<tr><td>{esc(m['row_id'])}</td><td style='text-align:left'>{esc(states.get(m['row_id'], '')[:220])}</td>"
        f"<td>{esc(m['gold'])}</td><td>{esc(m['pred'])}</td><td>{m['confidence']:.2f}</td></tr>"
        for m in result.get("confident_misses", [])
    )
    return f"""
<section class="card">
  <div class="qhead"><h2>{esc(qid)}</h2>
    <span class="badge"><b style="background:var(--{tone})">{icon}</b>{esc(label)}</span></div>
  <p class="qtext">{esc(instructions)}</p>
  <div class="tiles">{tiles}</div>
  <div class="charts">
    <div><h3>Does stated confidence match reality?</h3>
      <div class="legend"><span><i style="background:var(--series-1);height:10px;border-radius:2px"></i>Observed accuracy</span>
      <span><i style="background:repeating-linear-gradient(90deg,var(--text-primary) 0 4px,transparent 4px 8px);height:2px"></i>Perfectly calibrated (bars below it are overconfident)</span></div>
      {reliability_svg(result["bins"])}</div>
    <div><h3>What each threshold buys you</h3>
      <div class="legend"><span><i style="background:var(--series-1)"></i>Accepted accuracy</span>
      <span><i style="background:var(--series-2)"></i>Handled by fast model</span></div>
      {coverage_svg(result["sweep"], result["threshold"], result["target"])}</div>
  </div>
  {notes}
  <details><summary>Accuracy by predicted answer, at the threshold</summary><div class="scroll">
    <table><tr><th>Predicted answer</th><th>Accepted</th><th>Accuracy</th></tr>{answers}</table></div></details>
  <details><summary>Confident and wrong: check these labels first ({len(result.get("confident_misses", []))})</summary><div class="scroll">
    <table><tr><th>Row</th><th style="text-align:left">State</th><th>Label</th><th>Model</th><th>Confidence</th></tr>{misses}</table></div>
    <p class="qtext" style="margin-top:8px">A confident miss is usually a wrong label or an ambiguous question, not a model failure. Fix those before trusting any threshold.</p></details>
  <details><summary>Threshold table</summary><div class="scroll">
    <table><tr><th>Threshold</th><th>Handled</th><th>Accepted accuracy</th><th>95% lower bound</th><th>Rows</th></tr>{sweep_rows}</table></div>
    <p class="qtext" style="margin-top:8px">How well each confidence measure separates right from wrong (AUROC): {auroc}</p></details>
</section>"""


def render(title: str, meta: dict, questions: dict, results: dict[str, dict], cost: dict | None,
           states: dict[str, str] | None = None) -> str:
    top = [tile("Labeled rows", f"{meta['n_rows']:,}", f"{pct(meta['holdout'], 0)} held out")]
    if cost:
        top += [
            tile("Rows that escalate", pct(cost["escalation_rate"]), "any question below its threshold"),
            tile("Cascade cost / 1k rows", f"${cost['per_1k']['cascade']:.3f}", f"LLM only ${cost['per_1k']['llm_only']:.2f}"),
            tile("Saved vs LLM only", pct(cost["savings_vs_llm_only"], 0), f"~{cost['latency_ms']['cascade']:.0f} ms mean latency"),
        ]
    sim_note = ('<p class="note">These numbers come from the built-in simulator, not from Jev. '
                'They demonstrate the pipeline only.</p>') if meta.get("provider") == "sim" else ""
    sections = "".join(question_section(qid, questions[qid].instructions, result, states or {}) for qid, result in results.items())
    models = ", ".join(meta.get("models") or []) or "unknown"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{esc(title)}</h1>
<p class="meta">Provider {esc(meta.get("provider"))} &middot; answered by {esc(models)} &middot; generated {esc(meta.get("created"))}</p>
<div class="tiles">{"".join(top)}</div>
{sim_note}
{sections}
<footer>Generated by jevcal {esc(meta.get("version"))}. Thresholds are picked on the fit split and verified on the held-out split.
Cost figures use the assumptions passed on the command line.</footer>
</main><div id="tip"></div><script>{JS}</script></body></html>"""


def write(path: str | Path, *args, **kwargs) -> None:
    Path(path).write_text(render(*args, **kwargs))
