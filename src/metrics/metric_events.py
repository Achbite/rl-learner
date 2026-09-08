"""Durable, source-preserving transport for immutable metric-event facts."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Callable, Iterable

import grpc

from proto.common import identity_pb2 as common_pb2
from proto.metrics import training_pb2 as training_metrics_pb2
from proto.training import training_pb2_grpc
from proto.metrics import catalog_pb2_grpc as metric_catalog_pb2_grpc
from proto.metrics import transport_pb2 as metric_transport_pb2
from proto.metrics import transport_pb2_grpc as metric_transport_pb2_grpc


from .registered_metrics import LocalMetricProjector
from .train_metrics import TrainMetricProducer

SOURCE_ROLES = {"aiserver", "learner"}

class MetricEventContractError(ValueError):
    """A metric batch violates the immutable event contract."""


def _same_message(left, right) -> bool:
    return left.SerializeToString(deterministic=True) == right.SerializeToString(
        deterministic=True
    )


def _copy_message(message):
    result = type(message)()
    result.CopyFrom(message)
    return result


def _has_field(message, name: str) -> bool:
    try:
        return bool(message.HasField(name))
    except (AttributeError, ValueError):
        return False


def _source_key(source: common_pb2.ServiceInstanceIdentity) -> str:
    if not source.component or not source.instance_id:
        raise MetricEventContractError("metric source identity is incomplete")
    if int(source.lifecycle_epoch) <= 0:
        raise MetricEventContractError("metric source epoch must be positive")
    return json.dumps(
        [source.component, source.instance_id, int(source.lifecycle_epoch)],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def validate_metric_batch(
    batch: metric_transport_pb2.MetricBatch,
    *,
    role: str,
    source: common_pb2.ServiceInstanceIdentity,
    previous_cursor: metric_transport_pb2.MetricBatchCursor,
) -> None:
    if role not in SOURCE_ROLES:
        raise MetricEventContractError("metric source role is invalid")
    if not _same_message(batch.source, source):
        raise MetricEventContractError("metric batch source identity mismatch")
    if not _same_message(previous_cursor.source, source):
        raise MetricEventContractError("metric cursor source identity mismatch")
    if int(batch.batch_sequence) != int(
        previous_cursor.acknowledged_batch_sequence
    ) + 1:
        raise MetricEventContractError("metric batch sequence is not contiguous")
    if int(batch.created_at_unix_ms) <= 0:
        raise MetricEventContractError("metric batch created_at is invalid")
    previous_event = int(previous_cursor.acknowledged_event_sequence)
    if batch.events:
        if batch.heartbeat or batch.HasField("gap"):
            raise MetricEventContractError("event batch shape is invalid")
        sequences = [int(event.event_sequence) for event in batch.events]
        if sequences != list(range(previous_event + 1, previous_event + 1 + len(sequences))):
            raise MetricEventContractError("metric event sequence is not contiguous")
        if (
            int(batch.first_event_sequence) != sequences[0]
            or int(batch.last_event_sequence) != sequences[-1]
        ):
            raise MetricEventContractError("metric event batch bounds are invalid")
        for event in batch.events:
            if not _has_field(event, "observed_at_unix_ms"):
                raise MetricEventContractError("metric observation time is missing")
            if not event.fact_payload or int(event.fact_kind) <= 0:
                raise MetricEventContractError("metric payload envelope is incomplete")
        next_event = sequences[-1]
    elif batch.HasField("gap"):
        gap = batch.gap
        if batch.heartbeat or not gap.reason:
            raise MetricEventContractError("metric gap batch shape is invalid")
        first = int(gap.first_unavailable_event_sequence)
        last = int(gap.last_unavailable_event_sequence)
        oldest = int(gap.oldest_available_event_sequence)
        if not (first == previous_event + 1 <= last < oldest):
            raise MetricEventContractError("metric sequence gap is invalid")
        if (
            int(batch.first_event_sequence) != first
            or int(batch.last_event_sequence) != last
        ):
            raise MetricEventContractError("metric gap bounds are invalid")
        next_event = last
    else:
        if not batch.heartbeat:
            raise MetricEventContractError("empty metric batch is not a heartbeat")
        if int(batch.first_event_sequence) or int(batch.last_event_sequence):
            raise MetricEventContractError("metric heartbeat bounds must be zero")
        next_event = previous_event

    if batch.source_final:
        if int(batch.final_event_sequence) != next_event:
            raise MetricEventContractError(
                "metric final_event_sequence differs from committed history"
            )
    elif int(batch.final_event_sequence):
        raise MetricEventContractError(
            "non-final metric batch has final_event_sequence"
        )


def cursor_for_batch(
    batch: metric_transport_pb2.MetricBatch,
    previous_cursor: metric_transport_pb2.MetricBatchCursor,
) -> metric_transport_pb2.MetricBatchCursor:
    if batch.events or batch.HasField("gap"):
        event_sequence = int(batch.last_event_sequence)
    else:
        event_sequence = int(previous_cursor.acknowledged_event_sequence)
    return metric_transport_pb2.MetricBatchCursor(
        source=batch.source,
        acknowledged_batch_sequence=batch.batch_sequence,
        acknowledged_event_sequence=event_sequence,
    )


class RawMetricBatchStore:
    """SQLite journal retaining exact batch bytes and durable ACK cursors."""

    FORMAT_VERSION = 2

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._connection = sqlite3.connect(
            str(path), timeout=5.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._initialize()
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        except Exception:
            self._connection.close()
            raise

    def _initialize(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if tables and version != self.FORMAT_VERSION:
            raise MetricEventContractError(
                "metric journal uses an unsupported legacy storage format"
            )
        if not tables and version not in {0, self.FORMAT_VERSION}:
            raise MetricEventContractError(
                "metric journal storage format is unsupported"
            )
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metric_sources (
                    source_key TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    component TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    lifecycle_epoch TEXT NOT NULL,
                    committed_batch_sequence TEXT NOT NULL DEFAULT '0',
                    committed_event_sequence TEXT NOT NULL DEFAULT '0',
                    pending_batch_sequence TEXT,
                    pending_event_sequence TEXT,
                    final_acknowledged INTEGER NOT NULL DEFAULT 0,
                    incomplete INTEGER NOT NULL DEFAULT 0,
                    incomplete_reason TEXT NOT NULL DEFAULT '',
                    updated_at_unix_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metric_batches (
                    journal_id INTEGER PRIMARY KEY,
                    source_key TEXT NOT NULL,
                    batch_sequence TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'committed')),
                    payload BLOB NOT NULL,
                    persisted_at_unix_ms INTEGER NOT NULL,
                    acknowledged_at_unix_ms INTEGER,
                    UNIQUE(source_key, batch_sequence),
                    FOREIGN KEY(source_key) REFERENCES metric_sources(source_key)
                );
                CREATE TABLE IF NOT EXISTS metric_store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metric_export_consumers (
                    source_key TEXT PRIMARY KEY,
                    consumer_key TEXT NOT NULL,
                    consumer_component TEXT NOT NULL,
                    consumer_instance_id TEXT NOT NULL,
                    consumer_lifecycle_epoch TEXT NOT NULL,
                    committed_batch_sequence TEXT NOT NULL DEFAULT '0',
                    committed_event_sequence TEXT NOT NULL DEFAULT '0',
                    updated_at_unix_ms INTEGER NOT NULL,
                    FOREIGN KEY(source_key) REFERENCES metric_sources(source_key)
                );
                PRAGMA user_version = 2;
                """
            )

    @staticmethod
    def _row_source(row: sqlite3.Row) -> common_pb2.ServiceInstanceIdentity:
        return common_pb2.ServiceInstanceIdentity(
            component=row["component"],
            instance_id=row["instance_id"],
            lifecycle_epoch=int(row["lifecycle_epoch"]),
        )

    @staticmethod
    def _row_cursor(
        row: sqlite3.Row,
        *,
        pending: bool,
    ) -> metric_transport_pb2.MetricBatchCursor:
        prefix = "pending" if pending else "committed"
        batch_sequence = row[f"{prefix}_batch_sequence"]
        event_sequence = row[f"{prefix}_event_sequence"]
        return metric_transport_pb2.MetricBatchCursor(
            source=RawMetricBatchStore._row_source(row),
            acknowledged_batch_sequence=int(batch_sequence or 0),
            acknowledged_event_sequence=int(event_sequence or 0),
        )

    @staticmethod
    def _decode_stored_batch(payload: bytes) -> metric_transport_pb2.MetricBatch:
        batch = metric_transport_pb2.MetricBatch()
        try:
            batch.ParseFromString(payload)
        except Exception as error:
            raise MetricEventContractError(
                "stored metric batch is not valid protobuf"
            ) from error
        if batch.SerializeToString(deterministic=True) != payload:
            raise MetricEventContractError(
                "stored metric batch bytes are not canonical"
            )
        return batch

    def activate_source(
        self,
        role: str,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> None:
        if role not in {"aiserver", "learner"}:
            raise MetricEventContractError("metric source role is invalid")
        key = _source_key(source)
        now_ms = int(time.time() * 1000)
        metadata_key = f"active_source:{role}"
        with self._lock, self._connection:
            active = self._connection.execute(
                "SELECT value FROM metric_store_metadata WHERE key = ?",
                (metadata_key,),
            ).fetchone()
            if active is not None and active["value"] != key:
                self._connection.execute(
                    """
                    UPDATE metric_sources
                    SET incomplete = 1,
                        incomplete_reason = CASE
                            WHEN incomplete_reason = ''
                            THEN 'source_replaced_before_final'
                            ELSE incomplete_reason
                        END,
                        updated_at_unix_ms = ?
                    WHERE source_key = ? AND final_acknowledged = 0
                    """,
                    (now_ms, active["value"]),
                )
            self._connection.execute(
                """
                INSERT INTO metric_sources(
                    source_key, role, component, instance_id,
                    lifecycle_epoch, updated_at_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    updated_at_unix_ms = excluded.updated_at_unix_ms
                """,
                (
                    key,
                    role,
                    source.component,
                    source.instance_id,
                    str(int(source.lifecycle_epoch)),
                    now_ms,
                ),
            )
            self._connection.execute(
                """
                INSERT INTO metric_store_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (metadata_key, key),
            )
            row = self._source_row(source)
            if row["role"] != role:
                raise MetricEventContractError(
                    "metric source was activated with another role"
                )

    def _source_row(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM metric_sources WHERE source_key = ?",
            (_source_key(source),),
        ).fetchone()
        if row is None:
            raise MetricEventContractError("metric source is not activated")
        return row

    def committed_cursor(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> metric_transport_pb2.MetricBatchCursor:
        with self._lock:
            return self._row_cursor(self._source_row(source), pending=False)

    def pending_cursor(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> metric_transport_pb2.MetricBatchCursor | None:
        with self._lock:
            row = self._source_row(source)
            if row["pending_batch_sequence"] is None:
                return None
            return self._row_cursor(row, pending=True)

    def pending_batch(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> metric_transport_pb2.MetricBatch | None:
        with self._lock:
            row = self._source_row(source)
            sequence = row["pending_batch_sequence"]
            if sequence is None:
                return None
            stored = self._connection.execute(
                """
                SELECT payload FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ? AND status = 'pending'
                """,
                (_source_key(source), str(int(sequence))),
            ).fetchone()
            if stored is None:
                raise MetricEventContractError(
                    "metric pending cursor has no raw batch"
                )
            batch = self._decode_stored_batch(stored["payload"])
            if (
                int(batch.batch_sequence) != int(sequence)
                or _source_key(batch.source) != _source_key(source)
            ):
                raise MetricEventContractError(
                    "stored pending batch identity is inconsistent"
                )
            return batch

    def persist_batch(
        self,
        role: str,
        batch: metric_transport_pb2.MetricBatch,
    ) -> metric_transport_pb2.MetricBatchCursor:
        self.activate_source(role, batch.source)
        key = _source_key(batch.source)
        now_ms = int(time.time() * 1000)
        with self._lock, self._connection:
            row = self._source_row(batch.source)
            committed = self._row_cursor(row, pending=False)
            pending = (
                None
                if row["pending_batch_sequence"] is None
                else self._row_cursor(row, pending=True)
            )
            candidate_cursor = cursor_for_batch(batch, committed)
            payload = batch.SerializeToString(deterministic=True)
            if pending is not None:
                stored_pending = self.pending_batch(batch.source)
                if (
                    _same_message(candidate_cursor, pending)
                    and stored_pending is not None
                    and stored_pending.SerializeToString(deterministic=True)
                    == payload
                ):
                    return pending
                raise MetricEventContractError(
                    "metric source already has an unacknowledged batch"
                )
            validate_metric_batch(
                batch,
                role=role,
                source=batch.source,
                previous_cursor=committed,
            )
            existing = self._connection.execute(
                """
                SELECT payload, status FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ?
                """,
                (key, str(int(batch.batch_sequence))),
            ).fetchone()
            if existing is not None:
                if existing["payload"] != payload:
                    raise MetricEventContractError(
                        "metric batch replay conflicts with durable bytes"
                    )
                if existing["status"] == "committed":
                    raise MetricEventContractError(
                        "metric batch replay is behind committed cursor"
                    )
            else:
                self._connection.execute(
                    """
                    INSERT INTO metric_batches(
                        source_key, batch_sequence, status, payload,
                        persisted_at_unix_ms
                    ) VALUES (?, ?, 'pending', ?, ?)
                    """,
                    (
                        key,
                        str(int(batch.batch_sequence)),
                        payload,
                        now_ms,
                    ),
                )
            self._connection.execute(
                """
                UPDATE metric_sources
                SET pending_batch_sequence = ?, pending_event_sequence = ?,
                    updated_at_unix_ms = ?
                WHERE source_key = ?
                """,
                (
                    str(int(candidate_cursor.acknowledged_batch_sequence)),
                    str(int(candidate_cursor.acknowledged_event_sequence)),
                    now_ms,
                    key,
                ),
            )
            self._changed.notify_all()
            return candidate_cursor

    def mark_acknowledged(
        self,
        batch: metric_transport_pb2.MetricBatch,
        cursor: metric_transport_pb2.MetricBatchCursor,
    ) -> None:
        key = _source_key(batch.source)
        now_ms = int(time.time() * 1000)
        with self._lock, self._connection:
            row = self._source_row(batch.source)
            if row["pending_batch_sequence"] is None:
                committed = self._row_cursor(row, pending=False)
                if _same_message(committed, cursor):
                    return
                raise MetricEventContractError("metric source has no pending batch")
            pending = self._row_cursor(row, pending=True)
            if not _same_message(pending, cursor):
                raise MetricEventContractError("metric ACK cursor is not pending")
            stored = self._connection.execute(
                """
                SELECT payload FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ? AND status = 'pending'
                """,
                (key, str(int(batch.batch_sequence))),
            ).fetchone()
            payload = batch.SerializeToString(deterministic=True)
            if (
                stored is None
                or stored["payload"] != payload
            ):
                raise MetricEventContractError(
                    "metric ACK does not identify durable raw bytes"
                )
            incomplete = int(row["incomplete"])
            incomplete_reason = str(row["incomplete_reason"])
            if batch.HasField("gap"):
                incomplete = 1
                if not incomplete_reason:
                    incomplete_reason = (
                        "sequence_gap:"
                        f"{batch.gap.first_unavailable_event_sequence}-"
                        f"{batch.gap.last_unavailable_event_sequence}"
                    )
            self._connection.execute(
                """
                UPDATE metric_batches
                SET status = 'committed', acknowledged_at_unix_ms = ?
                WHERE source_key = ? AND batch_sequence = ?
                """,
                (now_ms, key, str(int(batch.batch_sequence))),
            )
            self._connection.execute(
                """
                UPDATE metric_sources
                SET committed_batch_sequence = ?, committed_event_sequence = ?,
                    pending_batch_sequence = NULL,
                    pending_event_sequence = NULL,
                    final_acknowledged = ?, incomplete = ?,
                    incomplete_reason = ?, updated_at_unix_ms = ?
                WHERE source_key = ?
                """,
                (
                    str(int(cursor.acknowledged_batch_sequence)),
                    str(int(cursor.acknowledged_event_sequence)),
                    1 if batch.source_final else int(row["final_acknowledged"]),
                    incomplete,
                    incomplete_reason,
                    now_ms,
                    key,
                ),
            )
            self._changed.notify_all()

    def mark_incomplete(
        self,
        source: common_pb2.ServiceInstanceIdentity,
        reason: str,
    ) -> None:
        if not reason:
            raise ValueError("incomplete metric source requires a reason")
        with self._lock, self._connection:
            self._connection.execute(
                """
                UPDATE metric_sources
                SET incomplete = 1,
                    incomplete_reason = CASE
                        WHEN incomplete_reason = '' THEN ? ELSE incomplete_reason
                    END,
                    updated_at_unix_ms = ?
                WHERE source_key = ? AND final_acknowledged = 0
                """,
                (reason, int(time.time() * 1000), _source_key(source)),
            )

    def is_final(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> bool:
        with self._lock:
            return bool(self._source_row(source)["final_acknowledged"])

    def snapshot(self) -> dict:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT role, component, instance_id, lifecycle_epoch,
                       committed_batch_sequence, committed_event_sequence,
                       pending_batch_sequence, final_acknowledged,
                       incomplete, incomplete_reason
                FROM metric_sources
                ORDER BY role, component, instance_id, lifecycle_epoch
                """
            ).fetchall()
            batches = self._connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM metric_batches
                GROUP BY status
                """
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in batches}
        return {
            "enabled": True,
            "store": "sqlite-raw-batch",
            "committed_batch_count": counts.get("committed", 0),
            "pending_batch_count": counts.get("pending", 0),
            "incomplete_source_count": sum(
                1 for row in rows if bool(row["incomplete"])
            ),
            "sources": [
                {
                    "role": row["role"],
                    "component": row["component"],
                    "instance_id": row["instance_id"],
                    "lifecycle_epoch": int(row["lifecycle_epoch"]),
                    "committed_batch_sequence": int(
                        row["committed_batch_sequence"]
                    ),
                    "committed_event_sequence": int(
                        row["committed_event_sequence"]
                    ),
                    "pending_batch_sequence": (
                        None
                        if row["pending_batch_sequence"] is None
                        else int(row["pending_batch_sequence"])
                    ),
                    "final_acknowledged": bool(row["final_acknowledged"]),
                    "incomplete": bool(row["incomplete"]),
                    "incomplete_reason": row["incomplete_reason"],
                }
                for row in rows
            ],
        }

    def committed_batches_after(
        self, row_id: int
    ) -> list[tuple[int, str, str, metric_transport_pb2.MetricBatch]]:
        """Return durable committed batches in local persistence order."""
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
            raise ValueError("metric batch row_id must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT b.journal_id, b.source_key, s.role, b.payload
                FROM metric_batches AS b
                JOIN metric_sources AS s ON s.source_key = b.source_key
                WHERE b.status = 'committed' AND b.journal_id > ?
                ORDER BY b.journal_id
                """,
                (row_id,),
            ).fetchall()
        result = []
        for row in rows:
            batch = self._decode_stored_batch(row["payload"])
            if _source_key(batch.source) != str(row["source_key"]):
                raise MetricEventContractError(
                    "committed metric batch source is inconsistent"
                )
            result.append(
                (
                    int(row["journal_id"]),
                    str(row["role"]),
                    str(row["source_key"]),
                    batch,
                )
            )
        return result

    @staticmethod
    def _consumer_key(
        consumer: common_pb2.ServiceInstanceIdentity,
    ) -> str:
        return _source_key(consumer)

    def bind_export_consumer(
        self,
        source: common_pb2.ServiceInstanceIdentity,
        consumer: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        source_key = _source_key(source)
        consumer_key = self._consumer_key(consumer)
        now_ms = int(time.time() * 1000)
        with self._changed, self._connection:
            self._source_row(source)
            row = self._connection.execute(
                """
                SELECT consumer_key FROM metric_export_consumers
                WHERE source_key = ?
                """,
                (source_key,),
            ).fetchone()
            if row is not None:
                return str(row["consumer_key"]) == consumer_key
            self._connection.execute(
                """
                INSERT INTO metric_export_consumers(
                    source_key, consumer_key, consumer_component,
                    consumer_instance_id, consumer_lifecycle_epoch,
                    updated_at_unix_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_key,
                    consumer_key,
                    consumer.component,
                    consumer.instance_id,
                    str(int(consumer.lifecycle_epoch)),
                    now_ms,
                ),
            )
            self._changed.notify_all()
            return True

    def export_cursor(
        self,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> metric_transport_pb2.MetricBatchCursor:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT committed_batch_sequence, committed_event_sequence
                FROM metric_export_consumers WHERE source_key = ?
                """,
                (_source_key(source),),
            ).fetchone()
            if row is None:
                return metric_transport_pb2.MetricBatchCursor(source=source)
            return metric_transport_pb2.MetricBatchCursor(
                source=source,
                acknowledged_batch_sequence=int(
                    row["committed_batch_sequence"]
                ),
                acknowledged_event_sequence=int(
                    row["committed_event_sequence"]
                ),
            )

    def next_export_batch(
        self,
        source: common_pb2.ServiceInstanceIdentity,
        cursor: metric_transport_pb2.MetricBatchCursor,
    ) -> metric_transport_pb2.MetricBatch | None:
        if not _same_message(cursor.source, source):
            raise MetricEventContractError(
                "metric export cursor source identity mismatch"
            )
        next_sequence = int(cursor.acknowledged_batch_sequence) + 1
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ?
                  AND status = 'committed'
                """,
                (_source_key(source), str(next_sequence)),
            ).fetchone()
        if row is None:
            return None
        batch = self._decode_stored_batch(row["payload"])
        if (
            int(batch.batch_sequence) != next_sequence
            or _source_key(batch.source) != _source_key(source)
        ):
            raise MetricEventContractError(
                "stored learner metric export batch is corrupted"
            )
        return batch

    def export_availability(
        self,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> tuple[int, int, bool]:
        with self._lock:
            row = self._source_row(source)
            latest = int(row["committed_event_sequence"])
            return (1 if latest > 0 else 0, latest, bool(row["final_acknowledged"]))

    def acknowledge_export(
        self,
        source: common_pb2.ServiceInstanceIdentity,
        cursor: metric_transport_pb2.MetricBatchCursor,
    ) -> None:
        source_key = _source_key(source)
        with self._changed, self._connection:
            current = self.export_cursor(source)
            if _same_message(current, cursor):
                return
            batch = self.next_export_batch(source, current)
            if batch is None:
                raise MetricEventContractError(
                    "metric export ACK has no matching durable batch"
                )
            expected = cursor_for_batch(batch, current)
            if not _same_message(expected, cursor):
                raise MetricEventContractError(
                    "metric export ACK does not identify the next durable batch"
                )
            self._connection.execute(
                """
                UPDATE metric_export_consumers
                SET committed_batch_sequence = ?,
                    committed_event_sequence = ?,
                    updated_at_unix_ms = ?
                WHERE source_key = ?
                """,
                (
                    str(int(cursor.acknowledged_batch_sequence)),
                    str(int(cursor.acknowledged_event_sequence)),
                    int(time.time() * 1000),
                    source_key,
                ),
            )
            self._changed.notify_all()

    def wait_for_export_change(self, timeout: float) -> None:
        with self._changed:
            self._changed.wait(timeout=max(0.0, timeout))

    def wait_for_final_export_ack(
        self,
        source: common_pb2.ServiceInstanceIdentity,
        timeout: float,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._changed:
            while True:
                consumer = self._connection.execute(
                    """
                    SELECT committed_batch_sequence
                    FROM metric_export_consumers WHERE source_key = ?
                    """,
                    (_source_key(source),),
                ).fetchone()
                source_row = self._source_row(source)
                if consumer is None:
                    return False
                if bool(source_row["final_acknowledged"]) and int(
                    consumer["committed_batch_sequence"]
                ) == int(source_row["committed_batch_sequence"]):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._changed.wait(timeout=remaining)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class LearnerMetricEventService(
    metric_transport_pb2_grpc.MetricEventServiceServicer
):
    """Expose the Learner-owned raw journal to one exact consumer lifecycle."""

    def __init__(
        self,
        *,
        store: RawMetricBatchStore,
        source: common_pb2.ServiceInstanceIdentity,
    ):
        self.store = store
        self.source = _copy_message(source)

    @staticmethod
    def _valid_consumer(
        consumer: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        try:
            _source_key(consumer)
            return True
        except MetricEventContractError:
            return False

    def _fill_availability(self, response) -> None:
        response.producer.CopyFrom(self.source)
        oldest, latest, _ = self.store.export_availability(self.source)
        response.oldest_available_event_sequence = oldest
        response.latest_available_event_sequence = latest

    def GetMetricBatch(self, request, context):
        response = metric_transport_pb2.GetMetricBatchRsp()
        self._fill_availability(response)
        if not self._valid_consumer(request.consumer):
            response.result = metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID
            response.message = "metric consumer lifecycle identity is invalid"
            return response
        if not _same_message(request.cursor.source, self.source):
            response.result = metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_CURSOR
            response.message = "metric cursor source does not match producer"
            return response
        if not self.store.bind_export_consumer(
            self.source, request.consumer
        ):
            response.result = metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID
            response.message = "learner metric journal is pinned to another consumer"
            return response
        committed = self.store.export_cursor(self.source)
        if not _same_message(request.cursor, committed):
            response.result = metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_CURSOR
            response.message = "metric cursor does not match committed cursor"
            return response
        if (
            int(request.max_events) <= 0
            or int(request.max_events) > 1024
            or int(request.max_bytes) <= 0
            or int(request.max_bytes) > 16 * 1024 * 1024
            or int(request.wait_timeout_ms) < 0
            or int(request.wait_timeout_ms) > 5000
        ):
            response.result = metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID
            response.message = "metric batch limits are invalid"
            return response

        deadline = time.monotonic() + int(request.wait_timeout_ms) / 1000.0
        while True:
            batch = self.store.next_export_batch(self.source, committed)
            if batch is not None:
                if (
                    len(batch.events) > int(request.max_events)
                    or batch.ByteSize() > int(request.max_bytes)
                ):
                    response.result = (
                        metric_transport_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID
                    )
                    response.message = "requested limits are smaller than the next durable batch"
                    return response
                response.result = metric_transport_pb2.METRIC_BATCH_RESULT_DELIVERED
                response.message = "durable learner metric batch delivered"
                response.batch.CopyFrom(batch)
                self._fill_availability(response)
                return response
            _, _, source_final = self.store.export_availability(self.source)
            if source_final:
                response.result = metric_transport_pb2.METRIC_BATCH_RESULT_FINAL
                response.message = "learner metric source final batch is acknowledged"
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not context.is_active():
                response.result = metric_transport_pb2.METRIC_BATCH_RESULT_WAIT
                response.message = "no learner metric batch is currently available"
                return response
            self.store.wait_for_export_change(min(remaining, 0.25))

    def AckMetricBatch(self, request, context):
        del context
        response = metric_transport_pb2.AckMetricBatchRsp()
        self._fill_availability(response)
        if not self._valid_consumer(request.consumer) or not self.store.bind_export_consumer(
            self.source, request.consumer
        ):
            response.result = metric_transport_pb2.METRIC_BATCH_ACK_RESULT_REJECTED_INVALID
            response.message = "metric ACK consumer lifecycle is invalid"
            response.committed_cursor.CopyFrom(
                self.store.export_cursor(self.source)
            )
            return response
        if not _same_message(request.cursor.source, self.source):
            response.result = metric_transport_pb2.METRIC_BATCH_ACK_RESULT_REJECTED_CURSOR
            response.message = "metric ACK cursor source does not match producer"
            response.committed_cursor.CopyFrom(
                self.store.export_cursor(self.source)
            )
            return response
        committed = self.store.export_cursor(self.source)
        response.committed_cursor.CopyFrom(committed)
        if _same_message(request.cursor, committed):
            response.result = (
                metric_transport_pb2.METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED
            )
            response.message = "learner metric batch was already acknowledged"
            return response
        try:
            self.store.acknowledge_export(self.source, request.cursor)
        except MetricEventContractError as error:
            response.result = (
                metric_transport_pb2.METRIC_BATCH_ACK_RESULT_REJECTED_CURSOR
            )
            response.message = str(error)
            return response
        response.result = metric_transport_pb2.METRIC_BATCH_ACK_RESULT_APPLIED
        response.message = "learner metric batch acknowledged"
        response.committed_cursor.CopyFrom(
            self.store.export_cursor(self.source)
        )
        self._fill_availability(response)
        return response


def create_learner_metric_event_server(
    *,
    store: RawMetricBatchStore,
    source: common_pb2.ServiceInstanceIdentity,
    port: int,
    writer,
    status_snapshot,
):
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise MetricEventContractError(
            "learner metric event server port must be in [1, 65535]"
        )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    metric_transport_pb2_grpc.add_MetricEventServiceServicer_to_server(
        LearnerMetricEventService(
            store=store,
            source=source,
        ),
        server,
    )
    from .learner_status import LearnerStatusService, MetricCatalogService
    status = LearnerStatusService(source, status_snapshot)
    training_pb2_grpc.add_LearnerStatusServiceServicer_to_server(status, server)
    metric_catalog_pb2_grpc.add_MetricCatalogServiceServicer_to_server(MetricCatalogService(writer, status), server)
    bound = server.add_insecure_port(f"0.0.0.0:{port}")
    if bound != port:
        server.stop(0)
        raise MetricEventContractError(
            f"learner metric event server could not bind port {port}"
        )
    return server


class LocalTrainUpdateMetricWriter:
    """Persist committed Learner updates as one immutable local event each."""

    def __init__(
        self,
        store: RawMetricBatchStore,
        source: common_pb2.ServiceInstanceIdentity,
        initial_train_update_sequence: int = 0,
    ):
        self.store = store
        self.source = _copy_message(source)
        self.store.activate_source("learner", self.source)
        self._lock = threading.Lock()
        self._finalized = False
        self._producer = TrainMetricProducer()
        self._initial_train_update_sequence = int(
            initial_train_update_sequence
        )
        if self._initial_train_update_sequence < 0:
            raise MetricEventContractError(
                "initial train update sequence must be non-negative"
            )

    def catalog(self):
        return self._producer.registry.catalog(self.source)

    def _settle_pending(self) -> None:
        pending = self.store.pending_batch(self.source)
        cursor = self.store.pending_cursor(self.source)
        if pending is not None and cursor is not None:
            self.store.mark_acknowledged(pending, cursor)

    def _append_gap(
        self,
        *,
        first_event_sequence: int,
        last_event_sequence: int,
    ) -> None:
        if not 0 < first_event_sequence <= last_event_sequence:
            raise MetricEventContractError(
                "learner metric sequence gap bounds are invalid"
            )
        committed = self.store.committed_cursor(self.source)
        created_at_unix_ms = int(time.time() * 1000)
        if created_at_unix_ms <= 0:
            raise MetricEventContractError(
                "learner metric gap created_at must be positive"
            )
        batch = metric_transport_pb2.MetricBatch(
            source=self.source,
            batch_sequence=int(committed.acknowledged_batch_sequence) + 1,
            created_at_unix_ms=created_at_unix_ms,
            first_event_sequence=first_event_sequence,
            last_event_sequence=last_event_sequence,
            gap=metric_transport_pb2.MetricSequenceGap(
                first_unavailable_event_sequence=first_event_sequence,
                last_unavailable_event_sequence=last_event_sequence,
                oldest_available_event_sequence=last_event_sequence + 1,
                reason="learner_train_update_fact_unavailable",
            ),
        )
        cursor = self.store.persist_batch("learner", batch)
        self.store.mark_acknowledged(batch, cursor)

    def append(
        self,
        fact: training_metrics_pb2.TrainUpdateMetricFact,
        observed_at_unix_ms: int,
    ) -> None:
        with self._lock:
            if self._finalized:
                raise MetricEventContractError("learner metric source is final")
            self._settle_pending()
            committed = self.store.committed_cursor(self.source)
            expected_update_sequence = (
                self._initial_train_update_sequence
                + int(committed.acknowledged_event_sequence)
                + 1
            )
            actual_update_sequence = int(fact.train_update_sequence)
            if actual_update_sequence < expected_update_sequence:
                raise MetricEventContractError(
                    "train update sequence is behind learner event sequence"
                )
            if actual_update_sequence > expected_update_sequence:
                first_missing_event = (
                    int(committed.acknowledged_event_sequence) + 1
                )
                last_missing_event = (
                    actual_update_sequence
                    - self._initial_train_update_sequence
                    - 1
                )
                self._append_gap(
                    first_event_sequence=first_missing_event,
                    last_event_sequence=last_missing_event,
                )
                committed = self.store.committed_cursor(self.source)

            observed_at = int(observed_at_unix_ms)
            event_sequence = int(committed.acknowledged_event_sequence) + 1
            batch_sequence = int(committed.acknowledged_batch_sequence) + 1
            event = metric_transport_pb2.MetricEvent(
                event_sequence=event_sequence,
                observed_at_unix_ms=observed_at,
                fact_payload=self._producer.record(fact).SerializeToString(deterministic=True),
                fact_kind=metric_transport_pb2.METRIC_FACT_KIND_REGISTERED_METRICS,
            )
            batch = metric_transport_pb2.MetricBatch(
                source=self.source,
                batch_sequence=batch_sequence,
                created_at_unix_ms=int(time.time() * 1000),
                first_event_sequence=event_sequence,
                last_event_sequence=event_sequence,
                events=[event],
            )
            cursor = self.store.persist_batch("learner", batch)
            self.store.mark_acknowledged(batch, cursor)

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._settle_pending()
            committed = self.store.committed_cursor(self.source)
            finalized_at_unix_ms = int(time.time() * 1000)
            batch = metric_transport_pb2.MetricBatch(
                source=self.source,
                batch_sequence=int(committed.acknowledged_batch_sequence) + 1,
                created_at_unix_ms=finalized_at_unix_ms,
                heartbeat=True,
                source_final=True,
                final_event_sequence=int(
                    committed.acknowledged_event_sequence
                ),
            )
            cursor = self.store.persist_batch("learner", batch)
            self.store.mark_acknowledged(batch, cursor)
            self._finalized = True


class MetricEventCollector:
    """Pull producer batches and ACK only after durable local persistence."""

    GET_WAIT_TIMEOUT_MS = 5_000
    GET_RPC_TIMEOUT_SEC = 6.5
    ACK_RPC_TIMEOUT_SEC = 2.0
    INITIAL_RETRY_DELAY_SEC = 0.5
    MAX_RETRY_DELAY_SEC = 5.0

    def __init__(
        self,
        *,
        store: RawMetricBatchStore,
        consumer: common_pb2.ServiceInstanceIdentity,
        role: str,
        event_stub: metric_transport_pb2_grpc.MetricEventServiceStub,
        logger,
    ):
        self.store = store
        self.consumer = _copy_message(consumer)
        self.role = role
        self.event_stub = event_stub
        self.logger = logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_source: common_pb2.ServiceInstanceIdentity | None = None
        self._state_lock = threading.Lock()
        self._transport_state = "starting"
        self._transport_failure_count = 0
        self._transport_unavailable_since = 0.0
        self._transport_last_error = ""
        self._ever_connected = False

    def start(
        self, initial_source: common_pb2.ServiceInstanceIdentity
    ) -> "MetricEventCollector":
        if self._thread is not None:
            return self
        _source_key(initial_source)
        self._thread = threading.Thread(
            target=self._run,
            args=(_copy_message(initial_source),),
            name=f"{self.role}-metric-collector",
            daemon=True,
        )
        self._thread.start()
        return self

    @staticmethod
    def _rpc_error_text(error: grpc.RpcError) -> str:
        details = getattr(error, "details", None)
        if callable(details):
            try:
                value = details()
                if value:
                    return str(value)
            except Exception:
                pass
        return str(error)

    def _record_transport_unavailable(self, error: grpc.RpcError) -> None:
        now = time.monotonic()
        message = self._rpc_error_text(error)
        with self._state_lock:
            self._transport_failure_count += 1
            self._transport_last_error = message
            if self._transport_state in {"waiting", "unavailable"}:
                return
            self._transport_unavailable_since = now
            if self._ever_connected:
                self._transport_state = "unavailable"
                self.logger.warning(
                    "Producer metric relay became unavailable; training "
                    "continues and reconnect runs in background: %s",
                    message,
                )
            else:
                self._transport_state = "waiting"
                self.logger.info(
                    "Producer metric relay is waiting for Producer metric "
                    "service; training continues"
                )

    def _record_transport_connected(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            prior_state = self._transport_state
            if prior_state == "connected":
                return
            failure_count = self._transport_failure_count
            unavailable_since = self._transport_unavailable_since
            elapsed = (
                now - unavailable_since
                if unavailable_since > 0.0
                else None
            )
            if prior_state == "unavailable":
                self.logger.info(
                    "Producer metric relay recovered after %.1fs and %d "
                    "retry attempt(s)",
                    elapsed,
                    failure_count,
                )
            elif prior_state == "waiting":
                self.logger.info(
                    "Producer metric relay connected after waiting %.1fs "
                    "and %d retry attempt(s)",
                    elapsed,
                    failure_count,
                )
            else:
                self.logger.info("Producer metric relay connected")
            self._transport_state = "connected"
            self._transport_failure_count = 0
            self._transport_unavailable_since = 0.0
            self._transport_last_error = ""
            self._ever_connected = True

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "state": self._transport_state,
                "ever_connected": self._ever_connected,
                "retry_count": self._transport_failure_count,
                "last_error": self._transport_last_error,
            }

    def _ack_pending(
        self,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        batch = self.store.pending_batch(source)
        cursor = self.store.pending_cursor(source)
        if batch is None or cursor is None:
            return False
        response = self.event_stub.AckMetricBatch(
            metric_transport_pb2.AckMetricBatchReq(
                consumer=self.consumer,
                cursor=cursor,
            ),
            timeout=self.ACK_RPC_TIMEOUT_SEC,
        )
        positive = response.result in (
            metric_transport_pb2.METRIC_BATCH_ACK_RESULT_APPLIED,
            metric_transport_pb2.METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED,
        )
        if not positive:
            raise MetricEventContractError(
                response.message or "Producer metric ACK rejected"
            )
        if not _same_message(response.producer, source):
            raise MetricEventContractError("Producer metric ACK producer changed")
        if not _same_message(response.committed_cursor, cursor):
            raise MetricEventContractError(
                "Producer metric ACK committed another cursor"
            )
        self.store.mark_acknowledged(batch, cursor)
        return True

    def _pull_once(
        self,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        if self._ack_pending(source):
            return self.store.is_final(source)
        cursor = self.store.committed_cursor(source)
        response = self.event_stub.GetMetricBatch(
            metric_transport_pb2.GetMetricBatchReq(
                consumer=self.consumer,
                cursor=cursor,
                max_events=512,
                max_bytes=1024 * 1024,
                wait_timeout_ms=self.GET_WAIT_TIMEOUT_MS,
            ),
            timeout=self.GET_RPC_TIMEOUT_SEC,
        )
        positive = response.result in (
            metric_transport_pb2.METRIC_BATCH_RESULT_DELIVERED,
            metric_transport_pb2.METRIC_BATCH_RESULT_WAIT,
            metric_transport_pb2.METRIC_BATCH_RESULT_FINAL,
        )
        if not positive:
            raise MetricEventContractError(
                response.message or "Producer metric Get rejected"
            )
        if not _same_message(response.producer, source):
            raise MetricEventContractError("Producer metric producer changed")
        if response.result == metric_transport_pb2.METRIC_BATCH_RESULT_DELIVERED:
            if not response.HasField("batch"):
                raise MetricEventContractError(
                    "Producer delivered metric result without a batch"
                )
            batch = _copy_message(response.batch)
            if not _same_message(batch.source, source):
                raise MetricEventContractError(
                    "Producer delivered a batch from another source"
                )
            self.store.persist_batch(self.role, batch)
            self._ack_pending(source)
            return self.store.is_final(source)
        elif response.HasField("batch"):
            raise MetricEventContractError(
                "Producer non-delivery metric result contains a batch"
            )
        elif response.result == metric_transport_pb2.METRIC_BATCH_RESULT_FINAL:
            if not self.store.is_final(source):
                raise MetricEventContractError(
                    "Producer returned FINAL before local final ACK"
                )
            return True
        return False

    def _run(self, initial_source: common_pb2.ServiceInstanceIdentity) -> None:
        retry_delay = self.INITIAL_RETRY_DELAY_SEC
        while not self._stop.is_set():
            try:
                source = initial_source
                self.store.activate_source(self.role, source)
                self._active_source = source
                source_final = self._pull_once(source)
                self._record_transport_connected()
                retry_delay = self.INITIAL_RETRY_DELAY_SEC
                if source_final:
                    with self._state_lock:
                        self._transport_state = "final"
                    break
            except grpc.RpcError as error:
                self._record_transport_unavailable(error)
                self._stop.wait(retry_delay)
                retry_delay = min(
                    self.MAX_RETRY_DELAY_SEC, retry_delay * 2.0
                )
            except MetricEventContractError as error:
                self._record_terminal_failure("rejected", error)
                if self._active_source is not None:
                    try:
                        self.store.mark_incomplete(self._active_source, "metric_history_rejected")
                    except Exception as store_error:
                        self.logger.error("failed to mark rejected history: %s", store_error)
                break
            except Exception as error:
                self._record_terminal_failure("failed", error)
                break

    def _record_terminal_failure(self, state: str, error: Exception) -> None:
        with self._state_lock:
            self._transport_state = state
            self._transport_last_error = str(error)
        self.logger.error("Producer metric relay %s: %s", state, error)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._state_lock:
            if self._transport_state not in {"rejected", "failed", "final"}:
                self._transport_state = "stopped"
        if self._active_source is not None:
            try:
                if not self.store.is_final(self._active_source):
                    self.store.mark_incomplete(
                        self._active_source,
                        "relay_stopped_before_source_final",
                    )
            except Exception as error:
                self.logger.error(
                    "failed to record incomplete Producer metric source: %s",
                    error,
                )
