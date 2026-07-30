#!/usr/bin/env python3
"""Serve metrics for the currently active Learner training process."""

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
        self._records = []
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
                    self._records.append(
                        json.loads(raw_line.decode("utf-8"))
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    state["corrupt"] += 1
                    self._corrupt_lines += 1
        except OSError as exc:
            state["error"] = str(exc)

    def query(self, after_sequence: int = 0, limit: int = 0):
        self.refresh()
        with self._lock:
            records = [
                record
                for record in self._records
                if int(record.get("sequence", record.get("train_step", 0)))
                > after_sequence
            ]
            if limit > 0:
                records = records[:limit]
            return records

    def latest(self):
        self.refresh()
        with self._lock:
            return self._records[-1] if self._records else {}

    def status(self):
        latest = self.latest()
        with self._lock:
            timestamp = float(latest.get("timestamp", 0.0))
            interval_seconds = max(
                float(latest.get("interval_ms", 0.0)) / 1000.0, 0.0
            )
            stale_after = max(5.0, 3.0 * interval_seconds)
            age_seconds = (
                max(0.0, time.time() - timestamp) if timestamp else None
            )
            return {
                "schema_version": 1,
                "metrics_dir": self._metrics_dir,
                "mode": latest.get("mode", ""),
                "record_count": len(self._records),
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

    def summary(self):
        latest = self.latest()
        distributor = latest.get("distributor", {})
        rates = latest.get("rates", {})
        chain = latest.get("chain", {})
        return {
            "mode": latest.get("mode", ""),
            "sequence": latest.get("sequence", 0),
            "consumed": distributor.get("acked", 0),
            "consumer_sps": rates.get("trained_sps", 0.0),
            "queue_size": distributor.get("ready_samples", 0),
            "chain_ready": chain.get("ready", False),
        }


metrics_reader = None


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._json_response(
                {
                    "schema_version": 1,
                    "service": "learner-metrics",
                    "stream": "current",
                    "endpoints": [
                        "/api/metrics",
                        "/api/metrics/latest",
                        "/api/metrics/summary",
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
            records = metrics_reader.query(after_sequence, limit)
            self._json_response(
                {
                    "schema_version": 1,
                    "stream": "current",
                    "records": records,
                    "total": len(records),
                }
            )
        elif path == "/api/metrics/latest":
            self._json_response(
                {
                    "schema_version": 1,
                    "stream": "current",
                    "record": metrics_reader.latest(),
                }
            )
        elif path == "/api/metrics/summary":
            self._json_response(metrics_reader.summary())
        elif path == "/api/status":
            self._json_response(metrics_reader.status())
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
    parser = argparse.ArgumentParser(
        description="Serve the current training metrics API"
    )
    parser.add_argument("--dir", "-d", default="models/local-train/metrics")
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
