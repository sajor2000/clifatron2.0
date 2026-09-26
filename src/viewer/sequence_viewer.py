"""Token-sequence viewer (local browser app).

This is a lightweight inspection tool for generated CLIF-style token sequences
("clinical sentences"). It serves a small localhost web app that can load:
  • Parquet files (e.g., simulation outputs with a `generated_sequence` column)
  • Plain text files (one space-separated token sequence per line)

Run:
  uv run --frozen --group dev python -m src.viewer.sequence_viewer --parquet /path/to/sims.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import polars as pl
import pyarrow.parquet as pq

_ROW_IDX_COL = "__row_idx"

_SEQ_COL_CANDIDATES = [
    "generated_sequence",
    "sequence",
    "filtered_clif_text",
    "clif_text",
    "text",
    "tokens",
]

_ID_COL_CANDIDATES = [
    "simulation_id",
    "id",
    "row_id",
]

_META_COL_PREFER = [
    "hospitalization_id",
    "simulation_number",
    "dataset",
    "label_home",
    "label_ltach",
    "disposition",
    "task3_label",
    "task4_proportion",
]

_NUM = r"-?\d+(?:\.\d+)?"
_RANGE_RE = re.compile(rf"^(?P<concept>.+)_(?P<low>{_NUM})_(?P<high>{_NUM})$")
_VALUE_RE = re.compile(rf"^(?P<concept>.+)_(?P<value>{_NUM})$")

_DEFAULT_HIGHLIGHT_RE = re.compile(
    r"(disposition_|expired|death|mort|lactate|map|spo2|kdigo|pf|pao2|fio2|crrt|"
    r"vaso|norepi|epinephrine|dopamine|phenylephrine|vent|imv|intub|extub)",
    re.IGNORECASE,
)


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _safe_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _first_existing(candidates: list[str], columns: set[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def load_vocab_lock(path: Path) -> set[str]:
    blob = json.loads(path.read_text())
    vocab = blob.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"vocab_lock missing 'vocab' dict: {path}")
    return set(vocab.keys())


def parse_token(token: str) -> dict[str, Any]:
    """Best-effort parse of a fused clinical token."""
    raw = token
    if raw.startswith("[") and raw.endswith("]"):
        return {
            "raw": raw,
            "group": "special",
            "kind": "special",
            "concept": raw,
            "category": None,
            "low": None,
            "high": None,
            "value": None,
            "highlight": False,
        }

    parts = raw.split("_")
    group = parts[0] if parts else "other"

    m = _RANGE_RE.match(raw)
    if m is not None:
        concept = m.group("concept")
        return {
            "raw": raw,
            "group": group,
            "kind": "range",
            "concept": concept,
            "category": None,
            "low": float(m.group("low")),
            "high": float(m.group("high")),
            "value": None,
            "highlight": bool(_DEFAULT_HIGHLIGHT_RE.search(raw)),
        }

    m = _VALUE_RE.match(raw)
    if m is not None:
        concept = m.group("concept")
        return {
            "raw": raw,
            "group": group,
            "kind": "value",
            "concept": concept,
            "category": None,
            "low": None,
            "high": None,
            "value": float(m.group("value")),
            "highlight": bool(_DEFAULT_HIGHLIGHT_RE.search(raw)),
        }

    # Categorical or symbolic token: split last suffix off as "category".
    if len(parts) >= 2:
        concept = "_".join(parts[:-1])
        category = parts[-1]
    else:
        concept, category = raw, None

    return {
        "raw": raw,
        "group": group,
        "kind": "categorical",
        "concept": concept,
        "category": category,
        "low": None,
        "high": None,
        "value": None,
        "highlight": bool(_DEFAULT_HIGHLIGHT_RE.search(raw)),
    }


def summarize_sequence(tokens: list[str], *, vocab: set[str] | None = None) -> dict[str, Any]:
    parsed = [parse_token(t) for t in tokens]

    group_counts: dict[str, int] = {}
    concept_counts: dict[str, int] = {}
    for p in parsed:
        group_counts[p["group"]] = group_counts.get(p["group"], 0) + 1
        concept = p["concept"]
        concept_counts[concept] = concept_counts.get(concept, 0) + 1

    unknown = []
    if vocab is not None:
        for t in tokens:
            if t not in vocab:
                unknown.append(t)

    top_concepts = sorted(concept_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
    top_groups = sorted(group_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    return {
        "n_tokens": len(tokens),
        "n_unique": len(set(tokens)),
        "groups": top_groups,
        "top_concepts": top_concepts,
        "unknown": {
            "n": len(unknown),
            "rate": (len(unknown) / max(len(tokens), 1)),
            "sample": unknown[:50],
        }
        if vocab is not None
        else None,
        "parsed_tokens": parsed,
    }


class SequenceSource:
    name: str
    kind: str
    n_rows: int

    def list_rows(self, *, offset: int, limit: int, search: str | None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_row(self, *, row_id: str) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class ParquetSource(SequenceSource):
    path: Path
    name: str
    kind: str = "parquet"

    def __post_init__(self):
        schema = pq.read_schema(self.path)
        self._columns = set(schema.names)
        self.seq_col = _first_existing(_SEQ_COL_CANDIDATES, self._columns)
        if self.seq_col is None:
            raise ValueError(
                f"No sequence column found in {self.path}. "
                f"Tried: {', '.join(_SEQ_COL_CANDIDATES)}"
            )
        self.id_col = _first_existing(_ID_COL_CANDIDATES, self._columns)

        meta = [c for c in _META_COL_PREFER if c in self._columns]
        extra = sorted((self._columns - set(meta) - {self.seq_col}) & set(_META_COL_PREFER))
        self.meta_cols = meta + extra

        pf = pq.ParquetFile(self.path)
        self.n_rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0

    def _base_lazy(self) -> pl.LazyFrame:
        cols = [self.seq_col]
        cols.extend(self.meta_cols)
        if self.id_col is not None:
            cols.append(self.id_col)
        return pl.scan_parquet(str(self.path)).with_row_index(_ROW_IDX_COL).select(cols + [_ROW_IDX_COL])

    def list_rows(self, *, offset: int, limit: int, search: str | None) -> list[dict[str, Any]]:
        lf = self._base_lazy()
        if search:
            lf = lf.filter(pl.col(self.seq_col).str.contains(search, literal=True))
        df = lf.slice(offset, limit).collect(streaming=True)
        out: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            seq = (row.get(self.seq_col) or "").strip()
            tokens = seq.split()
            preview = " ".join(tokens[:50])
            rid = str(row[self.id_col]) if self.id_col and (row.get(self.id_col) is not None) else str(row[_ROW_IDX_COL])
            meta = {k: row.get(k) for k in self.meta_cols if k in row}
            out.append({
                "id": rid,
                "row_idx": int(row[_ROW_IDX_COL]),
                "preview": preview,
                "n_tokens": len(tokens),
                "meta": meta,
            })
        return out

    def get_row(self, *, row_id: str) -> dict[str, Any]:
        lf = self._base_lazy()
        if self.id_col is not None:
            # simulation_id is typically int; accept numeric strings.
            try:
                want: Any = int(row_id)
            except ValueError:
                want = row_id
            lf = lf.filter(pl.col(self.id_col) == want)
        else:
            lf = lf.filter(pl.col(_ROW_IDX_COL) == int(row_id))

        df = lf.limit(1).collect(streaming=True)
        if df.height == 0:
            raise KeyError(row_id)
        row = df.to_dicts()[0]
        return row


@dataclass
class TextSource(SequenceSource):
    path: Path
    name: str
    kind: str = "text"

    def __post_init__(self):
        self._offsets: list[int] = []
        off = 0
        with self.path.open("rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self._offsets.append(off)
                off = f.tell()
        self.n_rows = len(self._offsets)

    def _read_line(self, idx: int) -> str:
        if idx < 0 or idx >= self.n_rows:
            raise KeyError(str(idx))
        with self.path.open("rb") as f:
            f.seek(self._offsets[idx])
            line = f.readline().decode("utf-8", errors="replace")
        return line.strip("\r\n")

    def list_rows(self, *, offset: int, limit: int, search: str | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        end = min(offset + limit, self.n_rows)
        with self.path.open("rb") as f:
            if offset < self.n_rows:
                f.seek(self._offsets[offset])
            idx = offset
            while idx < end:
                line = f.readline()
                if not line:
                    break
                seq = line.decode("utf-8", errors="replace").strip()
                if search and search not in seq:
                    idx += 1
                    continue
                tokens = seq.split()
                out.append({
                    "id": str(idx),
                    "row_idx": idx,
                    "preview": " ".join(tokens[:50]),
                    "n_tokens": len(tokens),
                    "meta": {},
                })
                idx += 1
        return out

    def get_row(self, *, row_id: str) -> dict[str, Any]:
        idx = int(row_id)
        seq = self._read_line(idx)
        return {
            _ROW_IDX_COL: idx,
            "sequence": seq,
        }


@dataclass(frozen=True)
class ViewerState:
    sources: dict[str, SequenceSource]
    vocab: set[str] | None = None


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Token Sequence Viewer</title>
    <style>
      :root { --bg: #0b1020; --panel: #111a33; --text: #e8eefc; --muted: #9bb0e5; --line: #22305a; }
      body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--text); }
      header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; gap: 10px; align-items: center; }
      header .title { font-weight: 650; }
      header select, header input, header button { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
      header input { width: 320px; }
      header button { cursor: pointer; }
      main { display: flex; height: calc(100vh - 58px); }
      #list { width: 440px; border-right: 1px solid var(--line); overflow: auto; }
      #detail { flex: 1; overflow: auto; }
      .row { padding: 12px 14px; border-bottom: 1px solid var(--line); cursor: pointer; }
      .row:hover { background: rgba(255,255,255,0.03); }
      .row .meta { color: var(--muted); font-size: 12px; margin-top: 6px; }
      .row .preview { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; opacity: 0.95; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .section { padding: 14px 16px; border-bottom: 1px solid var(--line); }
      .section h2 { margin: 0 0 10px 0; font-size: 14px; color: var(--muted); font-weight: 600; }
      .kv { display: grid; grid-template-columns: 180px 1fr; gap: 8px; font-size: 13px; }
      .kv div { padding: 6px 0; border-bottom: 1px dotted rgba(255,255,255,0.08); }
      .kv code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
      #tokens { padding: 12px 16px; line-height: 1.85; }
      .token { display: inline-block; margin: 2px 4px 2px 0; padding: 3px 6px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.10); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
      .token.highlight { border-color: rgba(255, 100, 100, 0.8); box-shadow: 0 0 0 1px rgba(255,100,100,0.20) inset; }
      .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10); font-size: 12px; color: var(--muted); }
      .controls { display:flex; gap: 10px; align-items: center; margin-left: auto; }
      .muted { color: var(--muted); }
      .small { font-size: 12px; }
    </style>
  </head>
  <body>
    <header>
      <div class="title">Token Sequence Viewer</div>
      <select id="source"></select>
      <input id="search" placeholder="filter sequences (substring)" />
      <button id="refresh">Refresh</button>
      <div class="controls">
        <button id="prev">Prev</button>
        <span id="page" class="pill"></span>
        <button id="next">Next</button>
      </div>
    </header>
    <main>
      <div id="list"></div>
      <div id="detail">
        <div class="section">
          <h2>How to use</h2>
          <div class="small muted">Select a source, then click a row to render tokens + summary.</div>
        </div>
      </div>
    </main>
    <script>
      const state = { sources: [], source: null, offset: 0, limit: 100, search: "" };

      function esc(s) {
        return (s ?? "").toString().replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      }

      function colorForGroup(group) {
        let h = 0;
        for (let i = 0; i < group.length; i++) h = Math.imul(31, h) + group.charCodeAt(i) | 0;
        const hue = Math.abs(h) % 360;
        return `hsl(${hue}, 70%, 25%)`;
      }

      async function api(path) {
        const r = await fetch(path);
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return await r.json();
      }

      function setPageBadge(total) {
        const page = document.getElementById('page');
        const lo = state.offset + 1;
        const hi = Math.min(state.offset + state.limit, total);
        page.textContent = `${lo}-${hi} / ${total}`;
      }

      async function loadSources() {
        const data = await api('/api/sources');
        state.sources = data.sources;
        const sel = document.getElementById('source');
        sel.innerHTML = '';
        for (const s of state.sources) {
          const opt = document.createElement('option');
          opt.value = s.name;
          opt.textContent = `${s.name} (${s.kind}, ${s.n_rows} rows)`;
          sel.appendChild(opt);
        }
        state.source = state.sources[0]?.name ?? null;
        if (state.source) sel.value = state.source;
      }

      function renderList(rows) {
        const el = document.getElementById('list');
        el.innerHTML = '';
        for (const row of rows) {
          const div = document.createElement('div');
          div.className = 'row';
          div.onclick = () => loadRecord(row.id);
          const meta = Object.entries(row.meta || {}).map(([k,v]) => `${k}=${v}`).join(' · ');
          div.innerHTML = `
            <div><span class="pill">id=${esc(row.id)}</span> <span class="pill">n=${row.n_tokens}</span></div>
            <div class="preview">${esc(row.preview)}</div>
            <div class="meta">${esc(meta)}</div>
          `;
          el.appendChild(div);
        }
      }

      function renderDetail(data) {
        const el = document.getElementById('detail');
        const summary = data.summary;
        const meta = data.meta || {};
        const unknown = summary.unknown;

        let unknownHtml = '<div class="muted small">(no vocab provided)</div>';
        if (unknown) {
          unknownHtml = `<div class="small">unknown tokens: <code>${unknown.n}</code> (${(unknown.rate*100).toFixed(2)}%)</div>`
            + (unknown.sample?.length ? `<div class="small muted">sample: <code>${esc(unknown.sample.slice(0, 10).join(' '))}</code></div>` : '');
        }

        const groups = (summary.groups || []).slice(0, 20).map(([g,c]) => `<span class="pill" style="border-color: rgba(255,255,255,0.15);">${esc(g)}=${c}</span>`).join(' ');
        const topConcepts = (summary.top_concepts || []).slice(0, 25).map(([c,n]) => `<div><code>${esc(c)}</code></div><div>${n}</div>`).join('');

        el.innerHTML = `
          <div class="section">
            <h2>Record</h2>
            <div class="kv">
              <div>source</div><div><code>${esc(data.source)}</code></div>
              <div>id</div><div><code>${esc(data.id)}</code></div>
              <div>tokens</div><div><code>${summary.n_tokens}</code> <span class="muted small">(unique: ${summary.n_unique})</span></div>
              <div>meta</div><div><code>${esc(JSON.stringify(meta))}</code></div>
            </div>
          </div>
          <div class="section">
            <h2>Groups</h2>
            <div>${groups || '<span class="muted small">(none)</span>'}</div>
          </div>
          <div class="section">
            <h2>Vocab coverage</h2>
            ${unknownHtml}
          </div>
          <div class="section">
            <h2>Top concepts</h2>
            <div class="kv">${topConcepts || '<div class="muted small">(none)</div>'}</div>
          </div>
          <div class="section">
            <h2>Tokens</h2>
            <div id="tokens"></div>
          </div>
        `;

        const tokEl = document.getElementById('tokens');
        tokEl.innerHTML = '';
        for (const t of summary.parsed_tokens || []) {
          const span = document.createElement('span');
          span.className = 'token' + (t.highlight ? ' highlight' : '');
          const group = (t.group || 'other');
          span.style.background = `rgba(255,255,255,0.03)`;
          span.style.borderColor = `rgba(255,255,255,0.10)`;
          span.style.boxShadow = `0 0 0 1px ${colorForGroup(group)}33 inset`;
          const tip = [
            `group=${group}`,
            `kind=${t.kind}`,
            `concept=${t.concept}`,
            t.category ? `category=${t.category}` : null,
            t.low != null ? `low=${t.low}` : null,
            t.high != null ? `high=${t.high}` : null,
            t.value != null ? `value=${t.value}` : null,
          ].filter(Boolean).join('\n');
          span.title = tip;
          span.textContent = t.raw;
          tokEl.appendChild(span);
        }
      }

      async function loadRows() {
        if (!state.source) return;
        const q = new URLSearchParams({
          source: state.source,
          offset: state.offset.toString(),
          limit: state.limit.toString(),
          search: state.search,
        });
        const data = await api('/api/rows?' + q.toString());
        renderList(data.rows || []);
        setPageBadge(data.total || 0);
      }

      async function loadRecord(id) {
        const q = new URLSearchParams({ source: state.source, id: id.toString() });
        const data = await api('/api/record?' + q.toString());
        renderDetail(data);
      }

      function bindUI() {
        document.getElementById('source').addEventListener('change', async (e) => {
          state.source = e.target.value;
          state.offset = 0;
          await loadRows();
        });
        document.getElementById('search').addEventListener('input', async (e) => {
          state.search = e.target.value || '';
          state.offset = 0;
          await loadRows();
        });
        document.getElementById('refresh').addEventListener('click', async () => {
          await loadRows();
        });
        document.getElementById('prev').addEventListener('click', async () => {
          state.offset = Math.max(0, state.offset - state.limit);
          await loadRows();
        });
        document.getElementById('next').addEventListener('click', async () => {
          const src = state.sources.find(s => s.name === state.source);
          const total = src?.n_rows ?? 0;
          if (state.offset + state.limit >= total) return;
          state.offset = state.offset + state.limit;
          await loadRows();
        });
      }

      (async function main() {
        await loadSources();
        bindUI();
        await loadRows();
      })();
    </script>
  </body>
</html>
"""


def _make_handler(state: ViewerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, status: HTTPStatus, body: bytes, *, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, _json_dumps(payload), content_type="application/json; charset=utf-8")

        def _send_text(self, text: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, text.encode("utf-8"), content_type="text/plain; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            try:
                if path == "/":
                    self._send(HTTPStatus.OK, _INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
                    return

                if path == "/api/sources":
                    self._send_json({
                        "sources": [
                            {"name": s.name, "kind": s.kind, "n_rows": s.n_rows}
                            for s in state.sources.values()
                        ]
                    })
                    return

                if path == "/api/rows":
                    source = (qs.get("source") or [""])[0]
                    if source not in state.sources:
                        self._send_json({"error": f"unknown source: {source}"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    offset = _safe_int((qs.get("offset") or [None])[0], 0)
                    limit = _safe_int((qs.get("limit") or [None])[0], 100)
                    search = (qs.get("search") or [""])[0].strip() or None
                    src = state.sources[source]
                    rows = src.list_rows(offset=offset, limit=min(max(limit, 1), 500), search=search)
                    self._send_json({"source": source, "offset": offset, "limit": limit, "total": src.n_rows, "rows": rows})
                    return

                if path == "/api/record":
                    source = (qs.get("source") or [""])[0]
                    rid = (qs.get("id") or [""])[0]
                    if source not in state.sources:
                        self._send_json({"error": f"unknown source: {source}"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    if not rid:
                        self._send_json({"error": "missing id"}, status=HTTPStatus.BAD_REQUEST)
                        return

                    src = state.sources[source]
                    row = src.get_row(row_id=rid)
                    seq = (
                        row.get(getattr(src, "seq_col", "generated_sequence"))
                        or row.get("generated_sequence")
                        or row.get("sequence")
                        or row.get("filtered_clif_text")
                        or ""
                    ).strip()
                    tokens = [t for t in seq.split() if t]
                    summary = summarize_sequence(tokens, vocab=state.vocab)
                    meta = {k: v for k, v in row.items() if k not in {getattr(src, "seq_col", "generated_sequence"), "generated_sequence", "sequence", "filtered_clif_text"}}
                    self._send_json({"source": source, "id": rid, "sequence": seq, "meta": meta, "summary": summary})
                    return

                self._send_text("not found", status=HTTPStatus.NOT_FOUND)
            except KeyError:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Local web app to inspect token sequences")
    ap.add_argument("--parquet", action="append", default=[], help="Parquet file with token sequences")
    ap.add_argument("--txt", action="append", default=[], help="Text file (one sequence per line)")
    ap.add_argument("--vocab-lock", default=None, help="Optional vocab_lock.json to compute OOV rate")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8042)
    ap.add_argument("--no-open", action="store_true", help="Do not auto-open the browser")
    args = ap.parse_args(argv)

    sources: dict[str, SequenceSource] = {}
    for p in args.parquet:
        path = Path(p).expanduser().resolve()
        src = ParquetSource(path=path, name=path.stem)
        if src.name in sources:
            src = ParquetSource(path=path, name=f"{path.stem}-{len(sources)}")
        sources[src.name] = src

    for t in args.txt:
        path = Path(t).expanduser().resolve()
        src = TextSource(path=path, name=path.stem)
        if src.name in sources:
            src = TextSource(path=path, name=f"{path.stem}-{len(sources)}")
        sources[src.name] = src

    if not sources:
        ap.error("Provide at least one --parquet or --txt input")

    vocab = load_vocab_lock(Path(args.vocab_lock).expanduser().resolve()) if args.vocab_lock else None
    state = ViewerState(sources=sources, vocab=vocab)

    handler = _make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Token Sequence Viewer running on {url}")
    for s in sources.values():
        print(f"  - {s.name}: {s.kind} ({s.n_rows} rows)")
    if vocab is not None:
        print(f"  vocab: {len(vocab):,} tokens")

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
