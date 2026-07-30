#!/usr/bin/env python3
"""Serve the run-aware training metrics API."""

import argparse
import glob
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse


class MetricsFileReader:
    def __init__(self, metrics_dir: str):
        self._metrics_dir = os.path.abspath(metrics_dir)
        self._lock = threading.Lock()
        self._records_by_run = {}
        self._files = {}
        self._corrupt_lines = 0
        self._last_scan_time = 0.0
        os.makedirs(self._metrics_dir, exist_ok=True)
        print(f"[MetricsServer] 监控目录: {self._metrics_dir}")

    def refresh(self):
        with self._lock:
            now = time.monotonic()
            if now - self._last_scan_time >= 0.5:
                self._last_scan_time = now
                for path in glob.glob(
                    os.path.join(self._metrics_dir, "metrics_*.jsonl")
                ):
                    self._files.setdefault(
                        path, {"offset": 0, "pending": b"", "corrupt": 0}
                    )
            for path, state in list(self._files.items()):
                self._read_file(path, state)

    def _read_file(self, path: str, state: dict):
        try:
            file_size = os.path.getsize(path)
            if file_size < state["offset"]:
                state["offset"] = 0
                state["pending"] = b""
            if file_size == state["offset"]:
                return
            with open(path, "rb") as stream:
                stream.seek(state["offset"])
                data = state["pending"] + stream.read()
                state["offset"] = stream.tell()
            lines = data.split(b"\n")
            state["pending"] = lines.pop()
            for raw_line in lines:
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                    run_id = str(record.get("run_id", "legacy"))
                    self._records_by_run.setdefault(run_id, []).append(record)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    state["corrupt"] += 1
                    self._corrupt_lines += 1
        except OSError as exc:
            state["error"] = str(exc)

    def _selected_run(self, requested: str = "") -> str:
        if requested:
            return requested
        latest_run = ""
        latest_timestamp = -1.0
        for run_id, records in self._records_by_run.items():
            if records:
                timestamp = float(records[-1].get("timestamp", 0.0))
                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_run = run_id
        return latest_run

    def query(self, run_id: str = "", after_sequence: int = 0, limit: int = 0):
        self.refresh()
        with self._lock:
            selected = self._selected_run(run_id)
            records = [
                record
                for record in self._records_by_run.get(selected, [])
                if int(record.get("sequence", record.get("train_step", 0)))
                > after_sequence
            ]
            if limit > 0:
                records = records[:limit]
            return selected, records

    def latest(self, run_id: str = ""):
        self.refresh()
        with self._lock:
            selected = self._selected_run(run_id)
            records = self._records_by_run.get(selected, [])
            return selected, records[-1] if records else {}

    def runs(self):
        self.refresh()
        with self._lock:
            result = []
            for run_id, records in self._records_by_run.items():
                latest = records[-1] if records else {}
                result.append(
                    {
                        "run_id": run_id,
                        "mode": latest.get("mode", "unknown"),
                        "record_count": len(records),
                        "latest_sequence": latest.get(
                            "sequence", latest.get("train_step", 0)
                        ),
                        "latest_timestamp": latest.get("timestamp", 0),
                    }
                )
            return sorted(
                result, key=lambda item: item["latest_timestamp"], reverse=True
            )

    def status(self, run_id: str = ""):
        self.refresh()
        with self._lock:
            selected = self._selected_run(run_id)
            records = self._records_by_run.get(selected, [])
            latest = records[-1] if records else {}
            timestamp = float(latest.get("timestamp", 0.0))
            interval_seconds = max(
                float(latest.get("interval_ms", 0.0)) / 1000.0, 0.0
            )
            stale_after = max(5.0, 3.0 * interval_seconds)
            age_seconds = max(0.0, time.time() - timestamp) if timestamp else None
            return {
                "schema_version": 1,
                "metrics_dir": self._metrics_dir,
                "run_id": selected,
                "mode": latest.get("mode", ""),
                "record_count": len(records),
                "latest_sequence": latest.get(
                    "sequence", latest.get("train_step", 0)
                ),
                "age_seconds": age_seconds,
                "stale_after_seconds": stale_after,
                "stale": age_seconds is None or age_seconds > stale_after,
                "corrupt_line_count": self._corrupt_lines,
                "file_count": len(self._files),
                "file_errors": {
                    os.path.basename(path): state["error"]
                    for path, state in self._files.items()
                    if state.get("error")
                },
            }

    def summary(self, run_id: str = ""):
        selected, latest = self.latest(run_id)
        return {
            "run_id": selected,
            "mode": latest.get("mode", ""),
            "sequence": latest.get("sequence", 0),
            "consumed": latest.get("total_consumed", 0),
            "consumer_sps": latest.get("consumer_sps", 0.0),
            "queue_size": latest.get("distributor_queue_size", 0),
            "chain_ready": latest.get("chain_ready", False),
        }


metrics_reader = None


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        run_id = params.get("run_id", [""])[0]

        if path == "/":
            self._json_response(
                {
                    "schema_version": 1,
                    "service": "learner-metrics",
                    "endpoints": [
                        "/api/metrics",
                        "/api/metrics/latest",
                        "/api/metrics/summary",
                        "/api/runs",
                        "/api/status",
                    ],
                }
            )
        elif path == "/api/metrics":
            try:
                after_sequence = int(
                    params.get(
                        "after_sequence", params.get("since", ["0"])
                    )[0]
                )
                limit = int(params.get("limit", ["0"])[0])
                if after_sequence < 0 or limit < 0:
                    raise ValueError
            except ValueError:
                self._json_response(
                    {"schema_version": 1, "error": "invalid query cursor"},
                    status=400,
                )
                return
            selected, records = metrics_reader.query(
                run_id, after_sequence, limit
            )
            self._json_response(
                {
                    "schema_version": 1,
                    "run_id": selected,
                    "records": records,
                    "total": len(records),
                }
            )
        elif path == "/api/metrics/latest":
            selected, record = metrics_reader.latest(run_id)
            self._json_response(
                {"schema_version": 1, "run_id": selected, "record": record}
            )
        elif path == "/api/metrics/summary":
            self._json_response(metrics_reader.summary(run_id))
        elif path == "/api/runs":
            self._json_response(
                {"schema_version": 1, "runs": metrics_reader.runs()}
            )
        elif path == "/api/status":
            self._json_response(metrics_reader.status(run_id))
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json_response(self, document, status=200):
        try:
            body = json.dumps(
                document, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self.send_header(
                "Content-Type", "application/json; charset=utf-8"
            )
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format_string, *args):
        if "200" not in str(args):
            super().log_message(format_string, *args)


def main():
    global metrics_reader
    parser = argparse.ArgumentParser(description="Serve the training metrics API")
    parser.add_argument("--dir", "-d", default="logs/metrics")
    parser.add_argument("--port", "-p", type=int, default=9005)
    args = parser.parse_args()
    metrics_reader = MetricsFileReader(args.dir)
    metrics_reader.refresh()

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MetricsHTTPHandler)
    print(f"[MetricsServer] http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
