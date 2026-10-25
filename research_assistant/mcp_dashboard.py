"""
Tiny live dashboard for the MCP server.

Watches logs/mcp_calls.jsonl and renders every tool call as it happens:
timestamp, tool name, args preview, latency, result summary, error (if any).

Run:
    ./.venv/bin/python mcp_dashboard.py

Then open http://127.0.0.1:8765 in a browser. Auto-refreshes every 2s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "mcp_calls.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

app = FastAPI(title="Agentic RAG — MCP Live Dashboard")


def _read_recent_calls(limit: int = 100) -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries: List[Dict[str, Any]] = []
    for line in lines[-limit:][::-1]:  # newest first
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


@app.get("/api/calls", response_class=JSONResponse)
def api_calls(limit: int = 100) -> Dict[str, Any]:
    calls = _read_recent_calls(limit)
    return {
        "count": len(calls),
        "log_path": str(LOG_PATH),
        "log_exists": LOG_PATH.exists(),
        "calls": calls,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MCP Live Dashboard</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      background: #0f172a;
      color: #e2e8f0;
      padding: 24px;
    }
    h1 { font-size: 20px; margin: 0 0 4px; }
    .subtitle { color: #94a3b8; font-size: 13px; margin-bottom: 20px; }
    .meta { display: flex; gap: 24px; color: #64748b; font-size: 12px; margin-bottom: 16px; }
    .meta span { display: inline-flex; align-items: center; gap: 6px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
    .dot.pulse { animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .card {
      background: #1e293b;
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 10px;
      border-left: 3px solid #22c55e;
    }
    .card.err { border-left-color: #ef4444; }
    .row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
    .tool { font-weight: 600; font-size: 15px; color: #f1f5f9; }
    .tool.err { color: #fca5a5; }
    .time { color: #94a3b8; font-size: 12px; font-variant-numeric: tabular-nums; }
    .latency {
      color: #93c5fd;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      background: #1e3a8a33;
      padding: 2px 8px;
      border-radius: 999px;
    }
    .args, .result {
      margin-top: 8px;
      font-size: 12px;
      color: #cbd5e1;
      font-family: "Menlo", "Consolas", monospace;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f172a;
      padding: 8px 10px;
      border-radius: 6px;
    }
    .label {
      color: #64748b;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin: 8px 0 3px;
    }
    .trace-id {
      color: #a78bfa;
      font-size: 11px;
      font-family: "Menlo", monospace;
    }
    .empty {
      text-align: center;
      padding: 48px 12px;
      color: #64748b;
    }
    .empty .hint { margin-top: 8px; font-size: 12px; color: #475569; }
  </style>
</head>
<body>
  <h1>Agentic RAG — MCP Live Dashboard</h1>
  <div class="subtitle">Every tool call your MCP server receives, in real time.</div>
  <div class="meta">
    <span><span class="dot pulse"></span> Auto-refresh every 2s</span>
    <span id="count">0 calls logged</span>
    <span id="path"></span>
  </div>
  <div id="feed"></div>

  <script>
    const feed = document.getElementById('feed');
    const countEl = document.getElementById('count');
    const pathEl = document.getElementById('path');

    function fmtTs(ts) {
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString();
    }
    function fmtLatency(ms) {
      if (ms == null) return '';
      if (ms < 1000) return ms.toFixed(0) + ' ms';
      return (ms / 1000).toFixed(1) + ' s';
    }

    async function refresh() {
      try {
        const res = await fetch('/api/calls?limit=100');
        const data = await res.json();
        countEl.textContent = data.count + ' calls logged';
        pathEl.textContent = data.log_path;
        if (!data.calls || data.calls.length === 0) {
          feed.innerHTML = `
            <div class="empty">
              <div>No MCP calls yet.</div>
              <div class="hint">Ask Claude to call a tool — e.g. index_status — and it will appear here.</div>
            </div>`;
          return;
        }
        feed.innerHTML = data.calls.map(c => {
          const errCls = c.error ? ' err' : '';
          const argsPretty = JSON.stringify(c.args_preview || {}, null, 2);
          const resultPretty = c.error
            ? c.error
            : JSON.stringify(c.result_summary || {}, null, 2);
          return `
            <div class="card${errCls}">
              <div class="row">
                <div class="tool${errCls}">${c.tool}${c.error ? ' — ERROR' : ''}</div>
                <div>
                  <span class="latency">${fmtLatency(c.latency_ms)}</span>
                  <span class="time">${fmtTs(c.ts)}</span>
                </div>
              </div>
              <div class="label">args</div>
              <div class="args">${escapeHTML(argsPretty)}</div>
              <div class="label">${c.error ? 'error' : 'result'}</div>
              <div class="result">${escapeHTML(resultPretty)}</div>
              ${c.trace_id ? `<div class="trace-id" style="margin-top:8px">trace ${c.trace_id}</div>` : ''}
            </div>
          `;
        }).join('');
      } catch (e) {
        console.error(e);
      }
    }
    function escapeHTML(s) {
      return String(s).replace(/[&<>]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[ch]));
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("[dashboard] Serving on http://127.0.0.1:8765")
    print(f"[dashboard] Watching log file: {LOG_PATH}")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
