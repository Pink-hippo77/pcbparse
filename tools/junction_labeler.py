#!/usr/bin/env python3
"""Junction manual annotation tool (local web app, stdlib + optional PIL).

Shows each wire-junction patch with a source-image context crop and lets the
reviewer assign one of the three Dataset300 classes:

    1  connected_cross        (导通交叉：导线交叉且电气连通)
    2  non_connected_cross    (非导通交叉：导线交叉但不连通)
    3  noise                  (噪声/非交叉干扰)
    4  skip                   (跳过，稍后再回看)

Progress is saved incrementally to ``junction_review_progress.csv`` next to the
template, so you can close and resume at any time.

Run:
    E:\\EDA挑战题目六\\.venv-train\\Scripts\\python.exe tools\\junction_labeler.py
then open http://127.0.0.1:8765 in a browser.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is optional for context crops
    Image = None


RUN_DIR = (
    r"E:\EDA挑战题目六\work\materialized\wire-junction-v3"
    r"\runs\5233aeb838db659970f0ce73480753a2f4a4e79366612878986d3bf18bca610e"
)
SOURCE_ROOT = r"E:\EDA挑战题目六\Dataset300\Dataset300"
TEMPLATE_NAME = "junction_review_template.csv"
PROGRESS_NAME = "junction_review_progress.csv"

LABELS = ("connected_cross", "non_connected_cross", "noise")
LABEL_CN = {
    "connected_cross": "导通交叉",
    "non_connected_cross": "非导通交叉",
    "noise": "噪声",
}


def load_rows(template: Path, progress: Path) -> list[dict[str, str]]:
    with template.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"template has no rows: {template}")
    if progress.is_file():
        with progress.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            by_id = {row.get("record_id"): dict(row) for row in reader}
        for row in rows:
            saved = by_id.get(row.get("record_id"))
            if saved is not None:
                row.update(saved)
    return rows


def save_rows(rows: list[dict[str, str]], progress: Path) -> None:
    fieldnames = list(rows[0].keys())
    temporary = progress.with_name(f".{progress.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, progress)


class AnnotationStore:
    def __init__(self, template: Path, progress: Path, reviewer: str) -> None:
        self.template = template
        self.progress = progress
        self.reviewer = reviewer
        self._lock = threading.RLock()
        self.rows = load_rows(template, progress)
        self.index = 0

    def _first_pending(self, start: int = 0) -> int:
        count = len(self.rows)
        for offset in range(count):
            index = (start + offset) % count
            status = str(self.rows[index].get("review_status", "")).strip().lower()
            if status in {"", "pending"}:
                return index
        return -1

    def state(self) -> dict[str, Any]:
        with self._lock:
            total = len(self.rows)
            pending = sum(
                1
                for row in self.rows
                if str(row.get("review_status", "")).strip().lower() in {"", "pending"}
            )
            labeled = sum(
                1
                for row in self.rows
                if str(row.get("review_status", "")).strip().lower()
                in {"labeled", "reviewed"}
                and str(row.get("reviewed_label", "")).strip()
            )
            skipped = sum(
                1
                for row in self.rows
                if str(row.get("review_status", "")).strip().lower() == "skipped"
            )
            index = self._first_pending(self.index)
            return {
                "total": total,
                "pending": pending,
                "labeled": labeled,
                "skipped": skipped,
                "index": index if index >= 0 else None,
            }

    def item(self, index: int | None = None) -> dict[str, Any] | None:
        with self._lock:
            if index is None:
                index = self._first_pending(self.index)
            if index is None or index < 0 or index >= len(self.rows):
                return None
            self.index = index
            row = dict(self.rows[index])
            row["_index"] = index
            row["_total"] = len(self.rows)
            return row

    def annotate(self, record_id: str, label: str, notes: str, *, skip: bool) -> dict[str, Any]:
        with self._lock:
            if label not in LABELS and not skip:
                raise ValueError(f"unknown label {label!r}")
            for row in self.rows:
                if row.get("record_id") != record_id:
                    continue
                if skip:
                    row["review_status"] = "skipped"
                    row["reviewed_label"] = ""
                    row["reviewer"] = self.reviewer
                    row["notes"] = (notes or "").strip()
                else:
                    row["review_status"] = "labeled"
                    row["reviewed_label"] = label
                    row["reviewer"] = self.reviewer
                    row["notes"] = (notes or "").strip()
                save_rows(self.rows, self.progress)
                next_index = self._first_pending(self.index + 1)
                return {"next_index": next_index if next_index >= 0 else None}
            raise KeyError(record_id)

    def patch_bytes(self, row: dict[str, str]) -> bytes | None:
        raw = str(row.get("patch_path", "")).strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.template.parent / path
        if not path.is_file():
            return None
        return path.read_bytes()

    def context_crop(self, row: dict[str, str], size: int = 480) -> bytes | None:
        if Image is None:
            return None
        raw_source = str(row.get("source_image", "")).strip()
        if not raw_source:
            return None
        source = Path(SOURCE_ROOT) / raw_source
        if not source.is_file():
            return None
        try:
            cx = float(row.get("center_x", 0))
            cy = float(row.get("center_y", 0))
        except (TypeError, ValueError):
            return None
        with Image.open(source) as image:
            width, height = image.size
            half = size // 2
            left = max(0, int(cx) - half)
            top = max(0, int(cy) - half)
            right = min(width, left + size)
            bottom = min(height, top + size)
            left = max(0, right - size)
            top = max(0, bottom - size)
            crop = image.crop((left, top, right, bottom))
            if crop.mode in ("RGBA", "P", "LA"):
                crop = crop.convert("RGB")
            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG", quality=82)
            return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    store: AnnotationStore

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_html(INDEX_HTML.encode("utf-8"))
            return
        if self.path == "/api/state":
            self._send_json(self.store.state())
            return
        if self.path.startswith("/api/item"):
            index = None
            if self.path.startswith("/api/item/"):
                try:
                    index = int(self.path.rsplit("/", 1)[1])
                except ValueError:
                    index = None
            row = self.store.item(index)
            if row is None:
                self._send_json({"done": True, "message": "所有候选均已标注完成"}, status=200)
                return
            patch = self.store.patch_bytes(row)
            context = self.store.context_crop(row)
            payload = {
                "record_id": row.get("record_id"),
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "candidate_types": row.get("candidate_types", ""),
                "suggested_labels": row.get("suggested_labels", ""),
                "review_status": row.get("review_status", ""),
                "reviewed_label": row.get("reviewed_label", ""),
                "notes": row.get("notes", ""),
                "index": row.get("_index"),
                "total": row.get("_total"),
                "patch_data_url": (
                    "data:image/png;base64," + base64.b64encode(patch).decode("ascii")
                    if patch is not None
                    else None
                ),
                "context_data_url": (
                    "data:image/jpeg;base64," + base64.b64encode(context).decode("ascii")
                    if context is not None
                    else None
                ),
                "center_x": row.get("center_x"),
                "center_y": row.get("center_y"),
            }
            self._send_json(payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/annotate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            record_id = str(data.get("record_id", ""))
            label = str(data.get("label", ""))
            notes = str(data.get("notes", ""))
            skip = bool(data.get("skip", False))
            result = self.store.annotate(record_id, label, notes, skip=skip)
            self._send_json(result)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=400)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Junction 人工标注工具</title>
<style>
  :root { --ok:#16a34a; --no:#dc2626; --noise:#d97706; --skip:#6b7280; }
  * { box-sizing: border-box; }
  body { font-family: "Microsoft YaHei", system-ui, sans-serif; margin: 0;
         background: #0f172a; color: #e2e8f0; }
  header { padding: 12px 20px; background: #1e293b; display: flex;
           align-items: center; justify-content: space-between; }
  header h1 { font-size: 18px; margin: 0; }
  #progress { font-size: 14px; color: #94a3b8; }
  #bar { height: 4px; background: #334155; }
  #bar > div { height: 100%; background: var(--ok); width: 0%; transition: width .2s; }
  main { display: flex; gap: 20px; padding: 20px; flex-wrap: wrap; }
  .panel { background: #1e293b; border-radius: 12px; padding: 16px; }
  .images { display: flex; gap: 16px; flex-wrap: wrap; }
  .images figure { margin: 0; text-align: center; }
  .images img { background: #fff; border-radius: 8px; }
  #patch { image-rendering: pixelated; width: 256px; height: 256px; }
  #context { max-width: 440px; max-height: 440px; }
  .meta { font-size: 14px; line-height: 1.8; }
  .meta b { color: #fbbf24; }
  .buttons { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 14px; }
  .buttons button { font-size: 15px; padding: 14px; border: 0; border-radius: 8px;
                    color: #fff; cursor: pointer; }
  .b1 { background: var(--ok); } .b2 { background: var(--no); }
  .b3 { background: var(--noise); } .b4 { background: var(--skip); }
  .nav { display: flex; gap: 10px; margin-top: 14px; }
  .nav button { flex: 1; padding: 10px; border: 1px solid #475569; border-radius: 8px;
                background: #334155; color: #e2e8f0; cursor: pointer; }
  input, textarea { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #475569;
                    background: #0f172a; color: #e2e8f0; margin-top: 6px; }
  .kbd { display: inline-block; border: 1px solid #64748b; border-radius: 4px;
         padding: 0 6px; font-size: 12px; margin-left: 4px; }
  #toast { position: fixed; bottom: 20px; right: 20px; background: #334155; color: #e2e8f0;
           padding: 10px 16px; border-radius: 8px; opacity: 0; transition: opacity .3s; }
  #done { display: none; font-size: 18px; text-align: center; padding: 60px 20px; }
</style>
</head>
<body>
<header>
  <h1>Junction 人工标注工具</h1>
  <div id="progress">加载中…</div>
</header>
<div id="bar"><div id="barfill"></div></div>
<div id="done">
  <p>🎉 所有候选均已标注完成</p>
  <p style="font-size:14px;color:#94a3b8">进度保存在 junction_review_progress.csv，可随时复核。</p>
</div>
<main id="main">
  <section class="panel images">
    <figure><img id="patch" alt="junction patch"><figcaption>补丁图（64×64）</figcaption></figure>
    <figure><img id="context" alt="context"><figcaption>原图上下文（中心：x, y）</figcaption></figure>
  </section>
  <section class="panel">
    <div class="meta" id="meta">…</div>
    <label>备注</label>
    <textarea id="notes" rows="2" placeholder="可选：记录歧义或观察"></textarea>
    <div class="buttons">
      <button class="b1" onclick="annotate('connected_cross')">1 · 导通交叉</button>
      <button class="b2" onclick="annotate('non_connected_cross')">2 · 非导通交叉</button>
      <button class="b3" onclick="annotate('noise')">3 · 噪声</button>
      <button class="b4" onclick="annotate('', true)">4 · 跳过</button>
    </div>
    <div class="nav">
      <button onclick="goto(-1)">← 上一个</button>
      <button onclick="goto(1)">下一个 →</button>
    </div>
    <p style="font-size:12px;color:#94a3b8">
      快捷键：1 导通交叉 · 2 非导通交叉 · 3 噪声 · 4 跳过 · ←/→ 导航 · N 下一个未标
    </p>
  </section>
</main>
<div id="toast"></div>
<script>
let current = null;
let state = null;

function toast(text) {
  const el = document.getElementById("toast");
  el.textContent = text;
  el.style.opacity = "1";
  setTimeout(() => { el.style.opacity = "0"; }, 1400);
}

async function api(path, options) {
  const response = await fetch(path, options);
  return response.json();
}

async function refreshState() {
  state = await api("/api/state");
  const pct = state.total ? Math.round((state.labeled + state.skipped) / state.total * 100) : 0;
  document.getElementById("barfill").style.width = pct + "%";
  document.getElementById("progress").textContent =
    `已标注 ${state.labeled} · 跳过 ${state.skipped} · 待标 ${state.pending} / 共 ${state.total}`;
  return state;
}

async function loadItem(index) {
  const url = index == null ? "/api/item" : "/api/item/" + index;
  const item = await api(url);
  if (item.done) {
    document.getElementById("done").style.display = "block";
    document.getElementById("main").style.display = "none";
    return;
  }
  document.getElementById("done").style.display = "none";
  document.getElementById("main").style.display = "";
  current = item;
  document.getElementById("patch").src = item.patch_data_url || "";
  document.getElementById("context").src = item.context_data_url || "";
  document.getElementById("meta").innerHTML =
    `<div>记录：<b>${item.record_id}</b></div>` +
    `<div>样本：<b>${item.sample_id}</b> · 划分：<b>${item.split}</b></div>` +
    `<div>中心：<b>${item.center_x}, ${item.center_y}</b></div>` +
    `<div>候选类型：<b>${item.candidate_types || "—"}</b></div>` +
    `<div>建议标签：<b>${item.suggested_labels || "—"}</b></div>` +
    `<div style="color:#94a3b8">${item.index + 1} / ${item.total}</div>`;
  document.getElementById("notes").value = item.notes || "";
  document.getElementById("notes").focus();
}

async function annotate(label, skip) {
  if (!current) return;
  const result = await api("/api/annotate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      record_id: current.record_id,
      label: label,
      skip: Boolean(skip),
      notes: document.getElementById("notes").value
    })
  });
  if (result.error) { toast("错误：" + result.error); return; }
  toast(skip ? "已跳过" : `已标注：${label}`);
  await refreshState();
  await loadItem(result.next_index);
}

async function goto(delta) {
  if (!current) return;
  let target = current.index + delta;
  if (target < 0) target = 0;
  if (target >= current.total) target = current.total - 1;
  await loadItem(target);
}

async function nextPending() {
  const s = await refreshState();
  await loadItem(s.index);
}

document.addEventListener("keydown", (event) => {
  if (event.target && event.target.tagName === "TEXTAREA") {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      goto(1);
    }
    return;
  }
  if (event.key === "1") annotate("connected_cross");
  else if (event.key === "2") annotate("non_connected_cross");
  else if (event.key === "3") annotate("noise");
  else if (event.key === "4") annotate("", true);
  else if (event.key === "ArrowLeft") goto(-1);
  else if (event.key === "ArrowRight") goto(1);
  else if (event.key.toLowerCase() === "n") nextPending();
});

(async function init() {
  await refreshState();
  await loadItem(state.index);
})();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=RUN_DIR, help="materialized wire-junction run directory")
    parser.add_argument("--source-root", default=SOURCE_ROOT, help="Dataset300 source image root")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reviewer", default=os.environ.get("USERNAME", "reviewer"))
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    template = run_dir / TEMPLATE_NAME
    progress = run_dir / PROGRESS_NAME
    if not template.is_file():
        print(f"template not found: {template}", file=__import__("sys").stderr)
        return 2
    global SOURCE_ROOT
    SOURCE_ROOT = str(Path(args.source_root).expanduser().resolve())
    store = AnnotationStore(template, progress, reviewer=args.reviewer)
    Handler.store = store
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"junction labeler ready: {url}")
    print(f"template: {template}")
    print(f"progress: {progress}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
