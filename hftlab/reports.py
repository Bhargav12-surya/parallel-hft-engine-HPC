"""Offline, escaped exports of actual results. No external report assets required."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from hftlab import APP_NAME
from hftlab.utilities import atomic_json


def _flatten(value: dict, prefix: str = "") -> list[tuple[str, object]]:
    rows = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            rows.extend(_flatten(item, name))
        else:
            rows.append((name, item))
    return rows


def export_report(result: dict, path: Path | str) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        return atomic_json(path, result)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric", "value"])
            for key, value in _flatten(result):
                text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                # Neutralize spreadsheet formulas in user-controlled strings.
                if isinstance(value, str) and text.lstrip().startswith(("=", "+", "-", "@")):
                    text = "'" + text
                if key.lstrip().startswith(("=", "+", "-", "@")):
                    key = "'" + key
                writer.writerow([key, text])
        return path
    if path.suffix.lower() == ".md":
        rows = [f"# {APP_NAME} experiment", "", "Offline synthetic research results.", "", "| Metric | Value |", "|---|---|"]
        for key, value in _flatten(result):
            clean = html.escape(str(value)).replace("|", "\\|").replace("\n", " ")
            clean_key = html.escape(key).replace("|", "\\|").replace("\n", " ")
            rows.append(f"| {clean_key} | {clean} |")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path
    if path.suffix.lower() != ".html":
        raise ValueError("Report extension must be .html, .json, .csv, or .md")
    rows = "".join(f"<tr><th>{html.escape(key)}</th><td><pre>{html.escape(json.dumps(value, indent=2, default=str) if isinstance(value, (list, dict)) else str(value))}</pre></td></tr>"
                   for key, value in _flatten(result))
    chart = ""
    comparisons = result.get("benchmark", result).get("engines", [])
    valid = [item for item in comparisons if item.get("status") == "COMPLETED"] if isinstance(comparisons, list) else []
    if valid:
        maximum = max(item["timing"]["median_seconds"] for item in valid)
        bars = []
        for index, item in enumerate(valid):
            y = index*42 + 12
            width = 430*item["timing"]["median_seconds"]/maximum
            bars.append(f'<text x="4" y="{y+17}" fill="#cad5e3">{html.escape(item["engine"])}</text>'
                        f'<rect x="110" y="{y}" width="{width:.2f}" height="24" fill="#22d3ee"/>'
                        f'<text x="{120+width:.2f}" y="{y+17}" fill="#cad5e3">{item["timing"]["median_seconds"]:.6f}s</text>')
        chart = f'<h2>Median native runtime</h2><svg role="img" aria-label="Engine native runtime comparison" viewBox="0 0 700 {len(valid)*42+24}">{"".join(bars)}</svg>'
    trading = result.get("ai_simulation", result.get("result", result))
    if isinstance(trading, dict):
        curve = trading.get("equity_curve", [])
        if len(curve) > 1:
            low = min(point["equity"] for point in curve)
            high = max(point["equity"] for point in curve)
            span = max(high-low, 1e-9)
            start, end = curve[0]["timestamp_ns"], curve[-1]["timestamp_ns"]
            points = " ".join(f'{30+640*(point["timestamp_ns"]-start)/max(1,end-start):.2f},{200-160*(point["equity"]-low)/span:.2f}' for point in curve)
            chart += f'<h2>Chronological portfolio equity</h2><p>Sampled from actual native event accounting. Range {low:,.4f}–{high:,.4f}; time is simulated nanoseconds.</p><svg role="img" aria-label="Portfolio equity across simulated time" viewBox="0 0 700 230"><path d="M30 20V200H680" stroke="#475569" fill="none"/><polyline points="{points}" fill="none" stroke="#34d399" stroke-width="2"/></svg>'
    if "rows" in result and result.get("kind") in ("weak", "strong"):
        chart += '<h2>' + html.escape(result["kind"].title()) + ' scaling</h2><table><tr><th>Workers</th><th>Engine</th><th>Median seconds</th><th>Speedup vs same-input sequential</th></tr>'
        for row in result["rows"]:
            for engine in row["benchmark"].get("engines", []):
                if engine.get("status") == "COMPLETED":
                    chart += f'<tr><td>{row["workers"]}</td><td>{html.escape(engine["engine"])}</td><td>{engine["timing"]["median_seconds"]:.6f}</td><td>{engine["speedup"]:.3f}</td></tr>'
        chart += '</table>'
    path.write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(APP_NAME)} research report</title><style>body{{font:15px system-ui,sans-serif;background:#0b111b;color:#cbd5e1;max-width:1100px;margin:40px auto;padding:20px}}h1{{color:#22d3ee}}p{{color:#94a3b8}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #263449;padding:10px;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0}}svg{{max-width:100%;background:#101a29;border-radius:8px}}</style>
<h1>{html.escape(APP_NAME)}</h1><p>Offline experiment report</p>{chart}<table>{rows}</table></html>''', encoding="utf-8")
    return path
