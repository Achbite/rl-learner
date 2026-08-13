"""Durable, source-preserving transport for immutable metric-event facts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

import grpc

from proto import common_pb2, training_pb2, training_pb2_grpc


SHA256 = re.compile(r"[a-f0-9]{64}")
METRIC_SCHEMA_ID = "maze.metrics.v2"
METRIC_SCHEMA_VERSION = 2


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


def _digest_message(batch: training_pb2.MetricBatch) -> str:
    canonical = training_pb2.MetricBatch()
    canonical.CopyFrom(batch)
    canonical.ClearField("batch_digest")
    return hashlib.sha256(
        canonical.SerializeToString(deterministic=True)
    ).hexdigest()


def _content_digest(hex_digest: str) -> common_pb2.ContentDigest:
    if SHA256.fullmatch(hex_digest) is None:
        raise MetricEventContractError("metric digest must be lower-case SHA-256")
    return common_pb2.ContentDigest(
        algorithm=common_pb2.DIGEST_ALGORITHM_SHA256,
        hex=hex_digest,
    )


class MetricSchemaCatalog:
    """Canonical field catalog and SchemaIdentity loaded from contract bytes."""

    def __init__(self, document: dict, canonical_digest: str):
        if document.get("catalog_schema") != "rl.metric-field-catalog.v1":
            raise MetricEventContractError("metric catalog format is unsupported")
        if document.get("schema_id") != METRIC_SCHEMA_ID:
            raise MetricEventContractError("metric catalog schema_id is invalid")
        if int(document.get("schema_version", 0)) != METRIC_SCHEMA_VERSION:
            raise MetricEventContractError("metric catalog schema_version is invalid")
        fields = document.get("fields")
        if not isinstance(fields, list) or not fields:
            raise MetricEventContractError("metric catalog fields are missing")
        identities: list[tuple[str, str]] = []
        by_fact: dict[str, set[str]] = {
            "agent_episode": set(),
            "train_update": set(),
        }
        for field in fields:
            if not isinstance(field, dict):
                raise MetricEventContractError("metric catalog field is invalid")
            fact = str(field.get("fact", ""))
            field_id = str(field.get("field_id", ""))
            if fact not in by_fact or not field_id:
                raise MetricEventContractError("metric catalog identity is invalid")
            if field.get("aggregation") != "raw_sum_count":
                raise MetricEventContractError(
                    "metric catalog aggregation must be raw_sum_count"
                )
            identities.append((fact, field_id))
            by_fact[fact].add(field_id)
        if identities != sorted(identities) or len(identities) != len(
            set(identities)
        ):
            raise MetricEventContractError(
                "metric catalog identities must be unique and sorted"
            )
        if any(not values for values in by_fact.values()):
            raise MetricEventContractError("metric catalog fact fields are empty")
        if SHA256.fullmatch(canonical_digest) is None:
            raise MetricEventContractError("metric catalog digest is invalid")
        self.document = document
        self.canonical_digest = canonical_digest
        self.fields_by_fact = by_fact

    @classmethod
    def load(cls, directory: Path) -> "MetricSchemaCatalog":
        catalog_path = directory / "maze.metrics.v2.json"
        digest_path = directory / "maze.metrics.v2.sha256"
        catalog_bytes = catalog_path.read_bytes()
        actual_digest = hashlib.sha256(catalog_bytes).hexdigest()
        declared_digest = digest_path.read_text(encoding="utf-8").strip()
        if declared_digest != actual_digest:
            raise MetricEventContractError("metric catalog digest mismatch")
        try:
            document = json.loads(catalog_bytes)
        except json.JSONDecodeError as error:
            raise MetricEventContractError(
                f"metric catalog JSON is invalid: {error}"
            ) from error
        if not isinstance(document, dict):
            raise MetricEventContractError("metric catalog must be an object")
        return cls(document, actual_digest)

    def schema_identity(self) -> common_pb2.SchemaIdentity:
        return common_pb2.SchemaIdentity(
            schema_id=METRIC_SCHEMA_ID,
            schema_version=METRIC_SCHEMA_VERSION,
            canonical_digest=_content_digest(self.canonical_digest),
        )


def default_metric_schema_directory() -> Path:
    configured = os.environ.get("RL_METRIC_SCHEMA_DIR", "")
    if configured:
        return Path(configured).resolve()
    repository = Path(__file__).resolve().parents[2]
    local = repository / "schemas"
    if (local / "maze.metrics.v2.json").is_file():
        return local
    sibling_contracts = repository.parent / "rl-contracts" / "schemas"
    return sibling_contracts


def _validate_sum_counts(
    values,
    allowed_fields: set[str],
    owner: str,
) -> None:
    field_ids = [item.field_id for item in values]
    if len(field_ids) != len(set(field_ids)):
        raise MetricEventContractError(f"{owner} has duplicate field_id values")
    for item in values:
        if item.field_id not in allowed_fields:
            raise MetricEventContractError(
                f"{owner} field_id is outside maze.metrics.v2: {item.field_id}"
            )
        if int(item.count) <= 0 or not math.isfinite(float(item.sum)):
            raise MetricEventContractError(
                f"{owner} raw sum/count is invalid: {item.field_id}"
            )


def _validate_event(
    event: training_pb2.MetricEvent,
    batch: training_pb2.MetricBatch,
    catalog: MetricSchemaCatalog,
) -> None:
    if not _same_message(event.contract, batch.contract):
        raise MetricEventContractError("metric event contract differs from batch")
    if not _same_message(event.schema_identity, batch.schema_identity):
        raise MetricEventContractError("metric event schema differs from batch")
    if not _same_message(event.source, batch.source):
        raise MetricEventContractError("metric event source differs from batch")
    if int(event.event_sequence) <= 0 or int(event.committed_at_unix_ms) <= 0:
        raise MetricEventContractError("metric event identity is invalid")
    fact = event.WhichOneof("fact")
    if fact == "episode":
        episode = event.episode
        if not (
            episode.task_id
            and episode.environment_instance_id
            and episode.episode_id
            and episode.agents
        ):
            raise MetricEventContractError("episode metric fact is incomplete")
        agent_ids = [int(agent.agent_id) for agent in episode.agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise MetricEventContractError("episode metric fact repeats agent_id")
        for agent in episode.agents:
            if not agent.termination_reason:
                raise MetricEventContractError(
                    "agent episode termination_reason is missing"
                )
            if not agent.behavior_model_lineage_id:
                raise MetricEventContractError(
                    "agent episode behavior lineage is missing"
                )
            if (
                int(agent.behavior_model_version_min)
                > int(agent.behavior_model_version_max)
            ):
                raise MetricEventContractError(
                    "agent episode behavior version range is invalid"
                )
            if not math.isfinite(float(agent.episode_return)):
                raise MetricEventContractError("agent episode return is non-finite")
            if int(agent.blocked_move_count) > int(agent.attempted_move_count):
                raise MetricEventContractError(
                    "agent episode blocked moves exceed attempted moves"
                )
            _validate_sum_counts(
                agent.reward_components,
                catalog.fields_by_fact["agent_episode"],
                "episode reward component",
            )
            if any(
                int(component.count) != int(agent.transition_count)
                for component in agent.reward_components
            ):
                raise MetricEventContractError(
                    "reward component count differs from agent transitions"
                )
            component_sum = sum(
                float(component.sum) for component in agent.reward_components
            )
            tolerance = max(
                1e-6, abs(float(agent.episode_return)) * 1e-6
            )
            if abs(component_sum - float(agent.episode_return)) > tolerance:
                raise MetricEventContractError(
                    "reward components do not conserve agent episode return"
                )
    elif fact == "train_update":
        update = event.train_update
        if not (
            update.train_update_id
            and int(update.train_update_sequence) > 0
            and update.delivery_id
            and update.published_model.model_lineage_id
            and update.behavior_model_lineage_id
            and int(update.actual_batch_size) > 0
        ):
            raise MetricEventContractError("train update metric fact is incomplete")
        if (
            int(update.behavior_model_version_min)
            > int(update.behavior_model_version_max)
        ):
            raise MetricEventContractError(
                "train update behavior version range is invalid"
            )
        _validate_sum_counts(
            update.ppo_statistics,
            catalog.fields_by_fact["train_update"],
            "PPO statistic",
        )
    else:
        raise MetricEventContractError("metric event fact is missing")


def validate_metric_batch(
    batch: training_pb2.MetricBatch,
    *,
    contract: common_pb2.ContractIdentity,
    catalog: MetricSchemaCatalog,
    source: common_pb2.ServiceInstanceIdentity,
    previous_cursor: training_pb2.MetricBatchCursor,
    previous_watermark_unix_ms: int = 0,
) -> None:
    schema = catalog.schema_identity()
    if not _same_message(batch.contract, contract):
        raise MetricEventContractError("metric batch contract identity mismatch")
    if not _same_message(batch.schema_identity, schema):
        raise MetricEventContractError("metric batch schema identity mismatch")
    if not _same_message(batch.source, source):
        raise MetricEventContractError("metric batch source identity mismatch")
    if not _same_message(previous_cursor.source, source):
        raise MetricEventContractError("metric cursor source identity mismatch")
    if int(batch.batch_sequence) != int(
        previous_cursor.acknowledged_batch_sequence
    ) + 1:
        raise MetricEventContractError("metric batch sequence is not contiguous")
    if (
        batch.batch_digest.algorithm
        != common_pb2.DIGEST_ALGORITHM_SHA256
        or SHA256.fullmatch(batch.batch_digest.hex) is None
        or _digest_message(batch) != batch.batch_digest.hex
    ):
        raise MetricEventContractError("metric batch digest is invalid")
    if int(batch.created_at_unix_ms) <= 0:
        raise MetricEventContractError("metric batch created_at is invalid")
    watermark = int(batch.event_time_watermark_unix_ms)
    if watermark < int(previous_watermark_unix_ms):
        raise MetricEventContractError("metric event-time watermark regressed")

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
            _validate_event(event, batch, catalog)
        if watermark < max(
            int(event.committed_at_unix_ms) for event in batch.events
        ):
            raise MetricEventContractError(
                "metric watermark precedes a committed event"
            )
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
    batch: training_pb2.MetricBatch,
    previous_cursor: training_pb2.MetricBatchCursor,
) -> training_pb2.MetricBatchCursor:
    if batch.events or batch.HasField("gap"):
        event_sequence = int(batch.last_event_sequence)
    else:
        event_sequence = int(previous_cursor.acknowledged_event_sequence)
    return training_pb2.MetricBatchCursor(
        source=batch.source,
        acknowledged_batch_sequence=batch.batch_sequence,
        acknowledged_event_sequence=event_sequence,
        acknowledged_batch_digest=batch.batch_digest,
    )


class RawMetricBatchStore:
    """SQLite journal retaining exact batch bytes and durable ACK cursors."""

    def __init__(
        self,
        path: Path,
        contract: common_pb2.ContractIdentity,
        catalog: MetricSchemaCatalog,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.contract = _copy_message(contract)
        self.catalog = catalog
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path), timeout=5.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._initialize()
        except Exception:
            self._connection.close()
            raise

    def _initialize(self) -> None:
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
                    committed_digest TEXT NOT NULL DEFAULT '',
                    committed_watermark_unix_ms INTEGER NOT NULL DEFAULT 0,
                    pending_batch_sequence TEXT,
                    pending_event_sequence TEXT,
                    pending_digest TEXT,
                    final_acknowledged INTEGER NOT NULL DEFAULT 0,
                    incomplete INTEGER NOT NULL DEFAULT 0,
                    incomplete_reason TEXT NOT NULL DEFAULT '',
                    updated_at_unix_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metric_batches (
                    source_key TEXT NOT NULL,
                    batch_sequence TEXT NOT NULL,
                    batch_digest TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'committed')),
                    payload BLOB NOT NULL,
                    persisted_at_unix_ms INTEGER NOT NULL,
                    committed_at_unix_ms INTEGER,
                    PRIMARY KEY(source_key, batch_sequence),
                    FOREIGN KEY(source_key) REFERENCES metric_sources(source_key)
                );
                CREATE TABLE IF NOT EXISTS metric_store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
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
    ) -> training_pb2.MetricBatchCursor:
        prefix = "pending" if pending else "committed"
        batch_sequence = row[f"{prefix}_batch_sequence"]
        event_sequence = row[f"{prefix}_event_sequence"]
        digest = row[f"{prefix}_digest"]
        cursor = training_pb2.MetricBatchCursor(
            source=RawMetricBatchStore._row_source(row),
            acknowledged_batch_sequence=int(batch_sequence or 0),
            acknowledged_event_sequence=int(event_sequence or 0),
        )
        if digest:
            cursor.acknowledged_batch_digest.CopyFrom(_content_digest(digest))
        return cursor

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
    ) -> training_pb2.MetricBatchCursor:
        with self._lock:
            return self._row_cursor(self._source_row(source), pending=False)

    def pending_cursor(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> training_pb2.MetricBatchCursor | None:
        with self._lock:
            row = self._source_row(source)
            if row["pending_batch_sequence"] is None:
                return None
            return self._row_cursor(row, pending=True)

    def pending_batch(
        self, source: common_pb2.ServiceInstanceIdentity
    ) -> training_pb2.MetricBatch | None:
        with self._lock:
            row = self._source_row(source)
            sequence = row["pending_batch_sequence"]
            if sequence is None:
                return None
            stored = self._connection.execute(
                """
                SELECT payload, batch_digest FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ? AND status = 'pending'
                """,
                (_source_key(source), str(int(sequence))),
            ).fetchone()
            if stored is None:
                raise MetricEventContractError(
                    "metric pending cursor has no raw batch"
                )
            batch = training_pb2.MetricBatch.FromString(stored["payload"])
            if (
                batch.batch_digest.hex != stored["batch_digest"]
                or _digest_message(batch) != stored["batch_digest"]
            ):
                raise MetricEventContractError("stored metric batch is corrupted")
            return batch

    def persist_batch(
        self,
        role: str,
        batch: training_pb2.MetricBatch,
    ) -> training_pb2.MetricBatchCursor:
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
            if pending is not None:
                if _same_message(candidate_cursor, pending):
                    return pending
                raise MetricEventContractError(
                    "metric source already has an unacknowledged batch"
                )
            validate_metric_batch(
                batch,
                contract=self.contract,
                catalog=self.catalog,
                source=batch.source,
                previous_cursor=committed,
                previous_watermark_unix_ms=int(
                    row["committed_watermark_unix_ms"]
                ),
            )
            payload = batch.SerializeToString(deterministic=True)
            existing = self._connection.execute(
                """
                SELECT batch_digest, payload, status FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ?
                """,
                (key, str(int(batch.batch_sequence))),
            ).fetchone()
            if existing is not None:
                if (
                    existing["batch_digest"] != batch.batch_digest.hex
                    or existing["payload"] != payload
                ):
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
                        source_key, batch_sequence, batch_digest, status,
                        payload, persisted_at_unix_ms
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        key,
                        str(int(batch.batch_sequence)),
                        batch.batch_digest.hex,
                        payload,
                        now_ms,
                    ),
                )
            self._connection.execute(
                """
                UPDATE metric_sources
                SET pending_batch_sequence = ?, pending_event_sequence = ?,
                    pending_digest = ?, updated_at_unix_ms = ?
                WHERE source_key = ?
                """,
                (
                    str(int(candidate_cursor.acknowledged_batch_sequence)),
                    str(int(candidate_cursor.acknowledged_event_sequence)),
                    candidate_cursor.acknowledged_batch_digest.hex,
                    now_ms,
                    key,
                ),
            )
            return candidate_cursor

    def mark_acknowledged(
        self,
        batch: training_pb2.MetricBatch,
        cursor: training_pb2.MetricBatchCursor,
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
                SELECT batch_digest, payload FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ? AND status = 'pending'
                """,
                (key, str(int(batch.batch_sequence))),
            ).fetchone()
            payload = batch.SerializeToString(deterministic=True)
            if (
                stored is None
                or stored["batch_digest"] != batch.batch_digest.hex
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
                SET status = 'committed', committed_at_unix_ms = ?
                WHERE source_key = ? AND batch_sequence = ?
                """,
                (now_ms, key, str(int(batch.batch_sequence))),
            )
            self._connection.execute(
                """
                UPDATE metric_sources
                SET committed_batch_sequence = ?, committed_event_sequence = ?,
                    committed_digest = ?, committed_watermark_unix_ms = ?,
                    pending_batch_sequence = NULL,
                    pending_event_sequence = NULL, pending_digest = NULL,
                    final_acknowledged = ?, incomplete = ?,
                    incomplete_reason = ?, updated_at_unix_ms = ?
                WHERE source_key = ?
                """,
                (
                    str(int(cursor.acknowledged_batch_sequence)),
                    str(int(cursor.acknowledged_event_sequence)),
                    cursor.acknowledged_batch_digest.hex,
                    int(batch.event_time_watermark_unix_ms),
                    1 if batch.source_final else int(row["final_acknowledged"]),
                    incomplete,
                    incomplete_reason,
                    now_ms,
                    key,
                ),
            )

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
                       committed_watermark_unix_ms,
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
            "store": "sqlite-raw-batch-v1",
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
                    "committed_watermark_unix_ms": int(
                        row["committed_watermark_unix_ms"]
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
    ) -> list[tuple[int, str, str, training_pb2.MetricBatch]]:
        """Return durable committed batches in local persistence order."""
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
            raise ValueError("metric batch row_id must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT b.rowid AS row_id, b.source_key, s.role, b.payload,
                       b.batch_digest
                FROM metric_batches AS b
                JOIN metric_sources AS s ON s.source_key = b.source_key
                WHERE b.status = 'committed' AND b.rowid > ?
                ORDER BY b.rowid
                """,
                (row_id,),
            ).fetchall()
        result = []
        for row in rows:
            batch = training_pb2.MetricBatch.FromString(row["payload"])
            if (
                batch.batch_digest.hex != row["batch_digest"]
                or _digest_message(batch) != row["batch_digest"]
            ):
                raise MetricEventContractError(
                    "committed metric batch is corrupted"
                )
            result.append(
                (
                    int(row["row_id"]),
                    str(row["role"]),
                    str(row["source_key"]),
                    batch,
                )
            )
        return result

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _empty_episode_statistics() -> dict:
    return {
        "environment_episode_count": 0,
        "agent_episode_count": 0,
        "successful_agent_count": 0,
        "any_success_environment_count": 0,
        "all_success_environment_count": 0,
        "return_sum": 0.0,
        "return_count": 0,
        "return_min": None,
        "return_max": None,
        "transition_sum": 0,
        "transition_agent_count": 0,
        "unique_cell_sum": 0,
        "unique_cell_agent_count": 0,
        "blocked_move_sum": 0,
        "attempted_move_sum": 0,
        "path_ratio_sum": 0.0,
        "path_ratio_count": 0,
        "reward_components": {},
        "termination_counts": {},
        "behavior_model_version_min": None,
        "behavior_model_version_max": None,
        "behavior_model_lineages": set(),
    }


def _empty_train_statistics() -> dict:
    return {
        "train_update_count": 0,
        "actual_batch_size_sum": 0,
        "latest_train_update_sequence": 0,
        "latest_model_version": 0,
        "latest_cumulative_trained_samples": 0,
        "ppo": {},
        "behavior_model_version_min": None,
        "behavior_model_version_max": None,
        "behavior_model_lineages": set(),
    }


def _merge_minimum(current, candidate):
    return candidate if current is None else min(current, candidate)


def _merge_maximum(current, candidate):
    return candidate if current is None else max(current, candidate)


def _merge_episode_statistics(target: dict, source: dict) -> None:
    for key in (
        "environment_episode_count",
        "agent_episode_count",
        "successful_agent_count",
        "any_success_environment_count",
        "all_success_environment_count",
        "return_count",
        "transition_sum",
        "transition_agent_count",
        "unique_cell_sum",
        "unique_cell_agent_count",
        "blocked_move_sum",
        "attempted_move_sum",
        "path_ratio_count",
    ):
        target[key] += source[key]
    target["return_sum"] += source["return_sum"]
    target["path_ratio_sum"] += source["path_ratio_sum"]
    if source["return_min"] is not None:
        target["return_min"] = _merge_minimum(
            target["return_min"], source["return_min"]
        )
        target["return_max"] = _merge_maximum(
            target["return_max"], source["return_max"]
        )
    for name, counts in source["reward_components"].items():
        item = target["reward_components"].setdefault(
            name, {"sum": 0.0, "transition_count": 0, "agent_count": 0}
        )
        item["sum"] += counts["sum"]
        item["transition_count"] += counts["transition_count"]
        item["agent_count"] += counts["agent_count"]
    for reason, count in source["termination_counts"].items():
        target["termination_counts"][reason] = (
            target["termination_counts"].get(reason, 0) + count
        )
    if source["behavior_model_version_min"] is not None:
        target["behavior_model_version_min"] = _merge_minimum(
            target["behavior_model_version_min"],
            source["behavior_model_version_min"],
        )
        target["behavior_model_version_max"] = _merge_maximum(
            target["behavior_model_version_max"],
            source["behavior_model_version_max"],
        )
    target["behavior_model_lineages"].update(
        source["behavior_model_lineages"]
    )


def _merge_train_statistics(target: dict, source: dict) -> None:
    target["train_update_count"] += source["train_update_count"]
    target["actual_batch_size_sum"] += source["actual_batch_size_sum"]
    if source["latest_train_update_sequence"] >= target["latest_train_update_sequence"]:
        for key in (
            "latest_train_update_sequence",
            "latest_model_version",
            "latest_cumulative_trained_samples",
        ):
            target[key] = source[key]
    for name, counts in source["ppo"].items():
        item = target["ppo"].setdefault(name, {"sum": 0.0, "count": 0})
        item["sum"] += counts["sum"]
        item["count"] += counts["count"]
    if source["behavior_model_version_min"] is not None:
        target["behavior_model_version_min"] = _merge_minimum(
            target["behavior_model_version_min"],
            source["behavior_model_version_min"],
        )
        target["behavior_model_version_max"] = _merge_maximum(
            target["behavior_model_version_max"],
            source["behavior_model_version_max"],
        )
    target["behavior_model_lineages"].update(
        source["behavior_model_lineages"]
    )


def _episode_event_statistics(
    fact: training_pb2.EpisodeMetricFact,
) -> dict:
    result = _empty_episode_statistics()
    result["environment_episode_count"] = 1
    successes = 0
    for agent in fact.agents:
        transitions = int(agent.transition_count)
        attempted = int(agent.attempted_move_count)
        blocked = int(agent.blocked_move_count)
        if blocked > attempted:
            raise MetricEventContractError(
                "agent episode blocked moves exceed attempted moves"
            )
        result["agent_episode_count"] += 1
        result["return_sum"] += float(agent.episode_return)
        result["return_count"] += 1
        result["return_min"] = _merge_minimum(
            result["return_min"], float(agent.episode_return)
        )
        result["return_max"] = _merge_maximum(
            result["return_max"], float(agent.episode_return)
        )
        result["transition_sum"] += transitions
        result["transition_agent_count"] += 1
        result["unique_cell_sum"] += int(agent.unique_cell_count)
        result["unique_cell_agent_count"] += 1
        result["blocked_move_sum"] += blocked
        result["attempted_move_sum"] += attempted
        result["termination_counts"][agent.termination_reason] = (
            result["termination_counts"].get(agent.termination_reason, 0) + 1
        )
        result["behavior_model_version_min"] = _merge_minimum(
            result["behavior_model_version_min"],
            int(agent.behavior_model_version_min),
        )
        result["behavior_model_version_max"] = _merge_maximum(
            result["behavior_model_version_max"],
            int(agent.behavior_model_version_max),
        )
        result["behavior_model_lineages"].add(
            agent.behavior_model_lineage_id
        )
        component_sum = 0.0
        for component in agent.reward_components:
            if int(component.count) != transitions:
                raise MetricEventContractError(
                    "reward component count differs from agent transitions"
                )
            item = result["reward_components"].setdefault(
                component.field_id,
                {"sum": 0.0, "transition_count": 0, "agent_count": 0},
            )
            item["sum"] += float(component.sum)
            item["transition_count"] += int(component.count)
            item["agent_count"] += 1
            component_sum += float(component.sum)
        tolerance = max(1e-6, abs(float(agent.episode_return)) * 1e-6)
        if abs(component_sum - float(agent.episode_return)) > tolerance:
            raise MetricEventContractError(
                "reward component sums do not conserve agent episode return"
            )
        if agent.success:
            successes += 1
            result["successful_agent_count"] += 1
            if int(agent.shortest_action_steps) > 0:
                result["path_ratio_sum"] += transitions / int(
                    agent.shortest_action_steps
                )
                result["path_ratio_count"] += 1
    result["any_success_environment_count"] = 1 if successes else 0
    result["all_success_environment_count"] = (
        1 if successes == len(fact.agents) else 0
    )
    return result


def _train_event_statistics(fact: training_pb2.TrainUpdateMetricFact) -> dict:
    result = _empty_train_statistics()
    result["train_update_count"] = 1
    result["actual_batch_size_sum"] = int(fact.actual_batch_size)
    result["latest_train_update_sequence"] = int(fact.train_update_sequence)
    result["latest_model_version"] = int(fact.published_model.model_version)
    result["latest_cumulative_trained_samples"] = int(
        fact.cumulative_trained_samples
    )
    result["behavior_model_version_min"] = int(
        fact.behavior_model_version_min
    )
    result["behavior_model_version_max"] = int(
        fact.behavior_model_version_max
    )
    result["behavior_model_lineages"].add(fact.behavior_model_lineage_id)
    for statistic in fact.ppo_statistics:
        result["ppo"][statistic.field_id] = {
            "sum": float(statistic.sum),
            "count": int(statistic.count),
        }
    return result


def _ratio(numerator, denominator):
    return None if not denominator else numerator / denominator


def _render_episode_statistics(
    raw: dict,
    *,
    status: str,
    window_kind: str,
    requested_size: int | None = None,
    start_unix_ms: int | None = None,
    end_unix_ms: int | None = None,
) -> dict:
    episodes = int(raw["environment_episode_count"])
    values = {}
    if episodes:
        values = {
            "mean_agent_return": _ratio(raw["return_sum"], raw["return_count"]),
            "min_agent_return": raw["return_min"],
            "max_agent_return": raw["return_max"],
            "agent_success_rate": _ratio(
                raw["successful_agent_count"], raw["agent_episode_count"]
            ),
            "any_success_rate": _ratio(
                raw["any_success_environment_count"], episodes
            ),
            "all_success_rate": _ratio(
                raw["all_success_environment_count"], episodes
            ),
            "mean_episode_step": _ratio(
                raw["transition_sum"], raw["transition_agent_count"]
            ),
            "mean_unique_cells": _ratio(
                raw["unique_cell_sum"], raw["unique_cell_agent_count"]
            ),
            "blocked_move_rate": _ratio(
                raw["blocked_move_sum"], raw["attempted_move_sum"]
            ),
            "path_ratio_mean": _ratio(
                raw["path_ratio_sum"], raw["path_ratio_count"]
            ),
            "reward_components": {},
        }
        for name, item in sorted(raw["reward_components"].items()):
            values["reward_components"][name] = {
                "episode_mean": _ratio(item["sum"], item["agent_count"]),
                "transition_mean": _ratio(
                    item["sum"], item["transition_count"]
                ),
            }
    raw_document = {
        key: value
        for key, value in raw.items()
        if key not in {"behavior_model_lineages"}
    }
    raw_document["behavior_model_lineages"] = sorted(
        raw["behavior_model_lineages"]
    )
    return {
        "status": "no_data" if not episodes else status,
        "window_kind": window_kind,
        "requested_size": requested_size,
        "complete_window": (
            None if requested_size is None else episodes >= requested_size
        ),
        "start_unix_ms": start_unix_ms,
        "end_unix_ms": end_unix_ms,
        "values": values,
        "raw": raw_document,
    }


def _render_train_statistics(
    raw: dict,
    *,
    status: str,
    window_kind: str,
    start_unix_ms: int | None = None,
    end_unix_ms: int | None = None,
) -> dict:
    count = int(raw["train_update_count"])
    values = {}
    if count:
        values = {
            "latest_train_update_sequence": int(
                raw["latest_train_update_sequence"]
            ),
            "latest_model_version": int(raw["latest_model_version"]),
            "latest_cumulative_trained_samples": int(
                raw["latest_cumulative_trained_samples"]
            ),
            "ppo": {
                name: {"mean": _ratio(item["sum"], item["count"])}
                for name, item in sorted(raw["ppo"].items())
            },
        }
    raw_document = {
        key: value
        for key, value in raw.items()
        if key not in {"behavior_model_lineages"}
    }
    raw_document["behavior_model_lineages"] = sorted(
        raw["behavior_model_lineages"]
    )
    return {
        "status": "no_data" if not count else status,
        "window_kind": window_kind,
        "start_unix_ms": start_unix_ms,
        "end_unix_ms": end_unix_ms,
        "values": values,
        "raw": raw_document,
    }


class LocalMetricProjector:
    """Single-run projection of durable raw facts; never merges Server Pods."""

    BUCKET_MS = 5_000
    EPISODE_WINDOWS = (25, 100)
    TIME_WINDOWS_MS = {
        "5s": 5_000,
        "1m": 60_000,
        "1h": 3_600_000,
        "24h": 86_400_000,
    }

    def __init__(self, store: RawMetricBatchStore):
        self.store = store
        self._lock = threading.RLock()
        self._last_row_id = 0
        self._view_revision = 0
        self._episode_recent = deque(maxlen=max(self.EPISODE_WINDOWS))
        self._episode_buckets: dict[int, dict] = {}
        self._episode_all = _empty_episode_statistics()
        self._episode_latest: dict | None = None
        self._episode_watermark = 0
        self._episode_source_keys: set[str] = set()
        self._train_buckets: dict[int, dict] = {}
        self._train_all = _empty_train_statistics()
        self._train_latest: dict | None = None
        self._train_watermark = 0
        self._train_source_keys: set[str] = set()

    @staticmethod
    def _bucket_start(timestamp_unix_ms: int) -> int:
        return (int(timestamp_unix_ms) // LocalMetricProjector.BUCKET_MS) * (
            LocalMetricProjector.BUCKET_MS
        )

    def _accept_episode(
        self, source_key: str, event: training_pb2.MetricEvent
    ) -> None:
        statistics = _episode_event_statistics(event.episode)
        self._episode_source_keys.add(source_key)
        self._episode_recent.append(statistics)
        self._episode_latest = statistics
        _merge_episode_statistics(self._episode_all, statistics)
        bucket = self._episode_buckets.setdefault(
            self._bucket_start(event.committed_at_unix_ms),
            _empty_episode_statistics(),
        )
        _merge_episode_statistics(bucket, statistics)

    def _accept_train(
        self, source_key: str, event: training_pb2.MetricEvent
    ) -> None:
        statistics = _train_event_statistics(event.train_update)
        self._train_source_keys.add(source_key)
        self._train_latest = statistics
        _merge_train_statistics(self._train_all, statistics)
        bucket = self._train_buckets.setdefault(
            self._bucket_start(event.committed_at_unix_ms),
            _empty_train_statistics(),
        )
        _merge_train_statistics(bucket, statistics)

    @staticmethod
    def _window_statistics(
        buckets: dict[int, dict],
        cut_unix_ms: int,
        duration_ms: int,
        factory: Callable[[], dict],
        merge: Callable[[dict, dict], None],
    ) -> dict:
        result = factory()
        start = cut_unix_ms - duration_ms
        for bucket_start, statistics in buckets.items():
            if start <= bucket_start < cut_unix_ms:
                merge(result, statistics)
        return result

    def _advance(self) -> None:
        batches = self.store.committed_batches_after(self._last_row_id)
        for row_id, role, source_key, batch in batches:
            for event in batch.events:
                fact = event.WhichOneof("fact")
                if role == "aiserver" and fact == "episode":
                    self._accept_episode(source_key, event)
                elif role == "learner" and fact == "train_update":
                    self._accept_train(source_key, event)
                else:
                    raise MetricEventContractError(
                        "metric fact owner differs from durable source role"
                    )
            if role == "aiserver":
                self._episode_watermark = max(
                    self._episode_watermark,
                    int(batch.event_time_watermark_unix_ms),
                )
            elif role == "learner":
                self._train_watermark = max(
                    self._train_watermark,
                    int(batch.event_time_watermark_unix_ms),
                )
            self._last_row_id = row_id
            self._view_revision += 1
        episode_retention_start = (
            self._episode_watermark
            - max(self.TIME_WINDOWS_MS.values())
            - self.BUCKET_MS
        )
        if episode_retention_start > 0:
            self._episode_buckets = {
                key: value
                for key, value in self._episode_buckets.items()
                if key >= episode_retention_start
            }
        train_retention_start = (
            self._train_watermark
            - max(self.TIME_WINDOWS_MS.values())
            - self.BUCKET_MS
        )
        if train_retention_start > 0:
            self._train_buckets = {
                key: value
                for key, value in self._train_buckets.items()
                if key >= train_retention_start
            }

    @staticmethod
    def _projection_status(store_snapshot: dict) -> str:
        if store_snapshot.get("incomplete_source_count", 0):
            return "incomplete"
        sources = store_snapshot.get("sources", [])
        if sources and all(source.get("final_acknowledged") for source in sources):
            return "final"
        return "provisional"

    def snapshot(self) -> dict:
        with self._lock:
            self._advance()
            store_snapshot = self.store.snapshot()
            status = self._projection_status(store_snapshot)
            episode_cut = self._bucket_start(self._episode_watermark)
            train_cut = self._bucket_start(self._train_watermark)

            episode_windows = {}
            recent = list(self._episode_recent)
            for size in self.EPISODE_WINDOWS:
                raw = _empty_episode_statistics()
                for item in recent[-size:]:
                    _merge_episode_statistics(raw, item)
                episode_windows[str(size)] = _render_episode_statistics(
                    raw,
                    status=status,
                    window_kind="completed_environment_episodes",
                    requested_size=size,
                )
            for label, duration in self.TIME_WINDOWS_MS.items():
                raw = self._window_statistics(
                    self._episode_buckets,
                    episode_cut,
                    duration,
                    _empty_episode_statistics,
                    _merge_episode_statistics,
                )
                episode_windows[label] = _render_episode_statistics(
                    raw,
                    status=status,
                    window_kind="event_time",
                    start_unix_ms=episode_cut - duration,
                    end_unix_ms=episode_cut,
                )
            episode_windows["all"] = _render_episode_statistics(
                self._episode_all,
                status=status,
                window_kind="run_to_date",
            )

            train_windows = {}
            for label, duration in self.TIME_WINDOWS_MS.items():
                raw = self._window_statistics(
                    self._train_buckets,
                    train_cut,
                    duration,
                    _empty_train_statistics,
                    _merge_train_statistics,
                )
                train_windows[label] = _render_train_statistics(
                    raw,
                    status=status,
                    window_kind="event_time",
                    start_unix_ms=train_cut - duration,
                    end_unix_ms=train_cut,
                )
            train_windows["all"] = _render_train_statistics(
                self._train_all,
                status=status,
                window_kind="run_to_date",
            )
            latest_episode = _render_episode_statistics(
                self._episode_latest or _empty_episode_statistics(),
                status=status,
                window_kind="latest_environment_episode",
            )
            latest_train = _render_train_statistics(
                self._train_latest or _empty_train_statistics(),
                status=status,
                window_kind="latest_train_update",
            )
            return {
                "schema_identity": {
                    "schema_id": METRIC_SCHEMA_ID,
                    "schema_version": METRIC_SCHEMA_VERSION,
                    "canonical_digest": self.store.catalog.canonical_digest,
                },
                "view_revision": self._view_revision,
                "status": status,
                "multi_server_aggregation_performed": False,
                "server_source_policy": "sequential_run_scoped_lifecycles",
                "server_source_count": len(self._episode_source_keys),
                "learner_source_count": len(self._train_source_keys),
                "episodes": {
                    "event_time_watermark_unix_ms": self._episode_watermark,
                    "latest": latest_episode,
                    "windows": episode_windows,
                },
                "train_updates": {
                    "event_time_watermark_unix_ms": self._train_watermark,
                    "latest": latest_train,
                    "windows": train_windows,
                },
            }


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
        self._initial_train_update_sequence = int(
            initial_train_update_sequence
        )
        if self._initial_train_update_sequence < 0:
            raise MetricEventContractError(
                "initial train update sequence must be non-negative"
            )

    def _settle_pending(self) -> None:
        pending = self.store.pending_batch(self.source)
        cursor = self.store.pending_cursor(self.source)
        if pending is not None and cursor is not None:
            self.store.mark_acknowledged(pending, cursor)

    def append(
        self,
        fact: training_pb2.TrainUpdateMetricFact,
        committed_at_unix_ms: int,
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
            if int(fact.train_update_sequence) != expected_update_sequence:
                raise MetricEventContractError(
                    "train update sequence differs from learner event sequence"
                )
            event_sequence = int(committed.acknowledged_event_sequence) + 1
            batch_sequence = int(committed.acknowledged_batch_sequence) + 1
            event = training_pb2.MetricEvent(
                contract=self.store.contract,
                schema_identity=self.store.catalog.schema_identity(),
                source=self.source,
                event_sequence=event_sequence,
                committed_at_unix_ms=int(committed_at_unix_ms),
                train_update=fact,
            )
            batch = training_pb2.MetricBatch(
                contract=self.store.contract,
                schema_identity=self.store.catalog.schema_identity(),
                source=self.source,
                batch_sequence=batch_sequence,
                created_at_unix_ms=int(time.time() * 1000),
                first_event_sequence=event_sequence,
                last_event_sequence=event_sequence,
                events=[event],
                event_time_watermark_unix_ms=int(committed_at_unix_ms),
            )
            batch.batch_digest.CopyFrom(_content_digest(_digest_message(batch)))
            cursor = self.store.persist_batch("learner", batch)
            self.store.mark_acknowledged(batch, cursor)

    def finalize(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._settle_pending()
            committed = self.store.committed_cursor(self.source)
            batch = training_pb2.MetricBatch(
                contract=self.store.contract,
                schema_identity=self.store.catalog.schema_identity(),
                source=self.source,
                batch_sequence=int(committed.acknowledged_batch_sequence) + 1,
                created_at_unix_ms=int(time.time() * 1000),
                heartbeat=True,
                source_final=True,
                final_event_sequence=int(
                    committed.acknowledged_event_sequence
                ),
                event_time_watermark_unix_ms=int(time.time() * 1000),
            )
            batch.batch_digest.CopyFrom(_content_digest(_digest_message(batch)))
            cursor = self.store.persist_batch("learner", batch)
            self.store.mark_acknowledged(batch, cursor)
            self._finalized = True


class AIServerMetricRelay:
    """Pull AIServer batches and ACK only after durable local persistence."""

    GET_WAIT_TIMEOUT_MS = 5_000
    GET_RPC_TIMEOUT_SEC = 6.5
    ACK_RPC_TIMEOUT_SEC = 2.0

    def __init__(
        self,
        *,
        store: RawMetricBatchStore,
        contract: common_pb2.ContractIdentity,
        consumer: common_pb2.ServiceInstanceIdentity,
        status_stub: training_pb2_grpc.AIServerTrainingStatusServiceStub,
        event_stub: training_pb2_grpc.MetricEventServiceStub,
        logger,
    ):
        self.store = store
        self.contract = _copy_message(contract)
        self.consumer = _copy_message(consumer)
        self.status_stub = status_stub
        self.event_stub = event_stub
        self.logger = logger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_source: common_pb2.ServiceInstanceIdentity | None = None

    def start(self) -> "AIServerMetricRelay":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="aiserver-metric-relay",
            daemon=True,
        )
        self._thread.start()
        return self

    def _discover_source(self) -> common_pb2.ServiceInstanceIdentity:
        status = self.status_stub.GetAIServerStatus(
            training_pb2.AIServerStatusReq(), timeout=1.5
        )
        if not _same_message(status.contract, self.contract):
            raise MetricEventContractError(
                "AIServer metric source contract identity mismatch"
            )
        source = _copy_message(status.aiserver)
        _source_key(source)
        return source

    def _ack_pending(
        self,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        batch = self.store.pending_batch(source)
        cursor = self.store.pending_cursor(source)
        if batch is None or cursor is None:
            return False
        response = self.event_stub.AckMetricBatch(
            training_pb2.AckMetricBatchReq(
                contract=self.contract,
                consumer=self.consumer,
                cursor=cursor,
            ),
            timeout=self.ACK_RPC_TIMEOUT_SEC,
        )
        positive = response.result in (
            training_pb2.METRIC_BATCH_ACK_RESULT_APPLIED,
            training_pb2.METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED,
        )
        if (int(response.ret_code) == 0) != positive:
            raise MetricEventContractError(
                "AIServer metric ACK ret_code/result mismatch"
            )
        if not positive:
            raise MetricEventContractError(
                response.message or "AIServer metric ACK rejected"
            )
        if not _same_message(response.producer, source):
            raise MetricEventContractError("AIServer metric ACK producer changed")
        if not _same_message(response.committed_cursor, cursor):
            raise MetricEventContractError(
                "AIServer metric ACK committed another cursor"
            )
        self.store.mark_acknowledged(batch, cursor)
        return True

    def _pull_once(
        self,
        source: common_pb2.ServiceInstanceIdentity,
    ) -> bool:
        if self._ack_pending(source):
            return False
        cursor = self.store.committed_cursor(source)
        response = self.event_stub.GetMetricBatch(
            training_pb2.GetMetricBatchReq(
                contract=self.contract,
                consumer=self.consumer,
                cursor=cursor,
                max_events=512,
                max_bytes=1024 * 1024,
                wait_timeout_ms=self.GET_WAIT_TIMEOUT_MS,
            ),
            timeout=self.GET_RPC_TIMEOUT_SEC,
        )
        positive = response.result in (
            training_pb2.METRIC_BATCH_RESULT_DELIVERED,
            training_pb2.METRIC_BATCH_RESULT_WAIT,
            training_pb2.METRIC_BATCH_RESULT_FINAL,
        )
        if (int(response.ret_code) == 0) != positive:
            raise MetricEventContractError(
                "AIServer metric Get ret_code/result mismatch"
            )
        if not positive:
            raise MetricEventContractError(
                response.message or "AIServer metric Get rejected"
            )
        if not _same_message(response.producer, source):
            raise MetricEventContractError("AIServer metric producer changed")
        if response.result == training_pb2.METRIC_BATCH_RESULT_DELIVERED:
            if not response.HasField("batch"):
                raise MetricEventContractError(
                    "AIServer delivered metric result without a batch"
                )
            batch = _copy_message(response.batch)
            if not _same_message(batch.source, source):
                raise MetricEventContractError(
                    "AIServer delivered a batch from another source"
                )
            self.store.persist_batch("aiserver", batch)
            self._ack_pending(source)
        elif response.HasField("batch"):
            raise MetricEventContractError(
                "AIServer non-delivery metric result contains a batch"
            )
        elif response.result == training_pb2.METRIC_BATCH_RESULT_FINAL:
            if not self.store.is_final(source):
                raise MetricEventContractError(
                    "AIServer returned FINAL before local final ACK"
                )
            return True
        return False

    def _run(self) -> None:
        retry_delay = 0.1
        while not self._stop.is_set():
            try:
                source = self._discover_source()
                if (
                    self._active_source is not None
                    and not _same_message(self._active_source, source)
                    and not self.store.is_final(self._active_source)
                ):
                    self.store.mark_incomplete(
                        self._active_source,
                        "source_replaced_before_final",
                    )
                self.store.activate_source("aiserver", source)
                self._active_source = source
                source_final = self._pull_once(source)
                retry_delay = 0.1
                if source_final:
                    self._stop.wait(1.0)
            except grpc.RpcError as error:
                self.logger.warning(
                    "AIServer metric relay RPC unavailable: %s",
                    error.details() or str(error),
                )
                self._stop.wait(retry_delay)
                retry_delay = min(5.0, retry_delay * 2.0)
            except MetricEventContractError as error:
                if self._active_source is not None:
                    try:
                        self.store.mark_incomplete(
                            self._active_source,
                            "metric_contract_rejected",
                        )
                    except Exception:
                        pass
                self.logger.error("AIServer metric relay rejected history: %s", error)
                self._stop.wait(retry_delay)
                retry_delay = min(5.0, retry_delay * 2.0)
            except Exception as error:
                self.logger.error("AIServer metric relay failed: %s", error)
                self._stop.wait(retry_delay)
                retry_delay = min(5.0, retry_delay * 2.0)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.GET_RPC_TIMEOUT_SEC + 1.0)
        if self._active_source is not None:
            try:
                if not self.store.is_final(self._active_source):
                    self.store.mark_incomplete(
                        self._active_source,
                        "relay_stopped_before_source_final",
                    )
            except Exception as error:
                self.logger.error(
                    "failed to record incomplete AIServer metric source: %s",
                    error,
                )
