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
from concurrent import futures
from pathlib import Path
from typing import Callable, Iterable

import grpc

from proto import (
    common_pb2,
    maze_metrics_pb2,
    training_metrics_pb2,
    training_pb2,
    training_pb2_grpc,
)


SHA256 = re.compile(r"[a-f0-9]{64}")
METRIC_SCHEMA_VERSION = 1
MAZE_METRIC_SCHEMA_ID = "maze.episode.metrics"
TRAINING_METRIC_SCHEMA_ID = "rl.training.metrics"


class MetricEventContractError(ValueError):
    """A metric batch violates the immutable event contract."""


def _same_message(left, right) -> bool:
    return left.SerializeToString(deterministic=True) == right.SerializeToString(
        deterministic=True
    )


def _same_contract(left, right) -> bool:
    return (
        bool(left.package_name)
        and left.package_name == right.package_name
        and left.package_version == right.package_version
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
    """Schema-owned fact codecs selected by MetricBatch.schema_identity."""

    FILES = {
        MAZE_METRIC_SCHEMA_ID: "maze.episode.metrics.json",
        TRAINING_METRIC_SCHEMA_ID: "training.metrics.json",
    }
    FACT_NAMES = {
        MAZE_METRIC_SCHEMA_ID: "agent_episode",
        TRAINING_METRIC_SCHEMA_ID: "train_update",
    }
    MESSAGE_TYPES = {
        MAZE_METRIC_SCHEMA_ID: maze_metrics_pb2.EpisodeMetricFact,
        TRAINING_METRIC_SCHEMA_ID: training_metrics_pb2.TrainUpdateMetricFact,
    }
    ROLE_SCHEMAS = {
        "aiserver": MAZE_METRIC_SCHEMA_ID,
        "learner": TRAINING_METRIC_SCHEMA_ID,
    }

    def __init__(self, entries: dict[str, tuple[dict, str]]):
        if set(entries) != set(self.FILES):
            raise MetricEventContractError("metric schema set is incomplete")
        self._documents: dict[str, dict] = {}
        self._digests: dict[str, str] = {}
        self._fields: dict[str, set[str]] = {}
        for schema_id, (document, canonical_digest) in entries.items():
            if document.get("catalog_schema") != "rl.metric-field-catalog.v1":
                raise MetricEventContractError(
                    "metric catalog format is unsupported"
                )
            if document.get("schema_id") != schema_id:
                raise MetricEventContractError(
                    "metric catalog schema_id is invalid"
                )
            if int(document.get("schema_version", 0)) != METRIC_SCHEMA_VERSION:
                raise MetricEventContractError(
                    "metric catalog schema_version is invalid"
                )
            fields = document.get("fields")
            if not isinstance(fields, list) or not fields:
                raise MetricEventContractError("metric catalog fields are missing")
            expected_fact = self.FACT_NAMES[schema_id]
            identities: list[tuple[str, str]] = []
            field_ids: set[str] = set()
            for field in fields:
                if not isinstance(field, dict):
                    raise MetricEventContractError(
                        "metric catalog field is invalid"
                    )
                fact = str(field.get("fact", ""))
                field_id = str(field.get("field_id", ""))
                if fact != expected_fact or not field_id:
                    raise MetricEventContractError(
                        "metric catalog identity is invalid"
                    )
                if field.get("aggregation") != "raw_sum_count":
                    raise MetricEventContractError(
                        "metric catalog aggregation must be raw_sum_count"
                    )
                identities.append((fact, field_id))
                field_ids.add(field_id)
            if identities != sorted(identities) or len(identities) != len(
                set(identities)
            ):
                raise MetricEventContractError(
                    "metric catalog identities must be unique and sorted"
                )
            if SHA256.fullmatch(canonical_digest) is None:
                raise MetricEventContractError("metric catalog digest is invalid")
            self._documents[schema_id] = document
            self._digests[schema_id] = canonical_digest
            self._fields[schema_id] = field_ids

    @classmethod
    def load(cls, directory: Path) -> "MetricSchemaCatalog":
        entries: dict[str, tuple[dict, str]] = {}
        for schema_id, filename in cls.FILES.items():
            catalog_bytes = (directory / filename).read_bytes()
            try:
                document = json.loads(catalog_bytes)
            except json.JSONDecodeError as error:
                raise MetricEventContractError(
                    f"metric catalog JSON is invalid: {error}"
                ) from error
            if not isinstance(document, dict):
                raise MetricEventContractError(
                    "metric catalog must be an object"
                )
            entries[schema_id] = (
                document,
                hashlib.sha256(catalog_bytes).hexdigest(),
            )
        return cls(entries)

    def schema_identity(self, schema_id: str) -> common_pb2.SchemaIdentity:
        if schema_id not in self._digests:
            raise MetricEventContractError("metric schema is unsupported")
        return common_pb2.SchemaIdentity(
            schema_id=schema_id,
            schema_version=METRIC_SCHEMA_VERSION,
            canonical_digest=_content_digest(self._digests[schema_id]),
        )

    def validate_identity(
        self, identity: common_pb2.SchemaIdentity
    ) -> str:
        schema_id = str(identity.schema_id)
        if schema_id not in self._digests or not _same_message(
            identity, self.schema_identity(schema_id)
        ):
            raise MetricEventContractError(
                "metric batch schema identity mismatch"
            )
        return schema_id

    def fields_for(self, schema_id: str) -> set[str]:
        try:
            return self._fields[schema_id]
        except KeyError as error:
            raise MetricEventContractError(
                "metric schema is unsupported"
            ) from error

    def schema_id_for_role(self, role: str) -> str:
        try:
            return self.ROLE_SCHEMAS[role]
        except KeyError as error:
            raise MetricEventContractError(
                "metric source role is invalid"
            ) from error

    def decode(self, identity: common_pb2.SchemaIdentity, payload: bytes):
        schema_id = self.validate_identity(identity)
        if not payload:
            raise MetricEventContractError("metric fact payload is missing")
        message = self.MESSAGE_TYPES[schema_id]()
        try:
            message.ParseFromString(payload)
        except Exception as error:
            raise MetricEventContractError(
                "metric fact payload does not match its schema"
            ) from error
        return schema_id, message

    def identities_document(self) -> dict[str, dict]:
        return {
            schema_id: {
                "schema_id": schema_id,
                "schema_version": METRIC_SCHEMA_VERSION,
                "canonical_digest": self._digests[schema_id],
            }
            for schema_id in sorted(self._digests)
        }


def default_metric_schema_directory() -> Path:
    configured = os.environ.get("RL_METRIC_SCHEMA_DIR", "")
    if configured:
        return Path(configured).resolve()
    repository = Path(__file__).resolve().parents[2]
    local = repository / "schemas"
    if all((local / filename).is_file() for filename in MetricSchemaCatalog.FILES.values()):
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
                f"{owner} field_id is outside its metric schema: {item.field_id}"
            )
        if int(item.count) <= 0 or not math.isfinite(float(item.sum)):
            raise MetricEventContractError(
                f"{owner} raw sum/count is invalid: {item.field_id}"
            )


def _validate_event(
    event: training_pb2.MetricEvent,
    schema_identity: common_pb2.SchemaIdentity,
    catalog: MetricSchemaCatalog,
) -> None:
    if int(event.event_sequence) <= 0 or not _has_field(
        event, "observed_at_unix_ms"
    ):
        raise MetricEventContractError("metric event identity is invalid")
    schema_id, fact = catalog.decode(schema_identity, event.fact_payload)
    if schema_id == MAZE_METRIC_SCHEMA_ID:
        episode = fact
        if not (
            episode.environment_instance_id
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
                not _has_field(agent, "minimum_behavior_model_step")
                or not _has_field(agent, "maximum_behavior_model_step")
                or int(agent.minimum_behavior_model_step)
                > int(agent.maximum_behavior_model_step)
            ):
                raise MetricEventContractError(
                    "agent episode behavior step range is invalid"
                )
            if not math.isfinite(float(agent.episode_return)):
                raise MetricEventContractError("agent episode return is non-finite")
            if int(agent.blocked_move_count) > int(agent.attempted_move_count):
                raise MetricEventContractError(
                    "agent episode blocked moves exceed attempted moves"
                )
            _validate_sum_counts(
                agent.reward_components,
                catalog.fields_for(MAZE_METRIC_SCHEMA_ID),
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
    elif schema_id == TRAINING_METRIC_SCHEMA_ID:
        update = fact
        if not (
            update.train_update_id
            and int(update.train_update_sequence) > 0
            and update.delivery_id
            and update.published_model.model_lineage_id
            and _has_field(update.published_model, "model_step")
            and update.behavior_model_lineage_id
            and int(update.actual_batch_size) > 0
        ):
            raise MetricEventContractError("train update metric fact is incomplete")
        if (
            not _has_field(update, "minimum_behavior_model_step")
            or not _has_field(update, "maximum_behavior_model_step")
            or int(update.minimum_behavior_model_step)
            > int(update.maximum_behavior_model_step)
            or int(update.published_model.model_step)
            != int(update.train_update_sequence)
        ):
            raise MetricEventContractError(
                "train update model step contract is invalid"
            )
        _validate_sum_counts(
            update.ppo_statistics,
            catalog.fields_for(TRAINING_METRIC_SCHEMA_ID),
            "PPO statistic",
        )
    else:
        raise MetricEventContractError("metric event schema is unsupported")


def validate_metric_batch(
    batch: training_pb2.MetricBatch,
    *,
    contract: common_pb2.ContractIdentity,
    catalog: MetricSchemaCatalog,
    source: common_pb2.ServiceInstanceIdentity,
    previous_cursor: training_pb2.MetricBatchCursor,
) -> None:
    catalog.validate_identity(batch.schema_identity)
    if not _same_contract(batch.contract, contract):
        raise MetricEventContractError("metric batch contract identity mismatch")
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
            _validate_event(event, batch.schema_identity, catalog)
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
        self._changed = threading.Condition(self._lock)
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
                    acknowledged_at_unix_ms INTEGER,
                    PRIMARY KEY(source_key, batch_sequence),
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
                    committed_digest TEXT NOT NULL DEFAULT '',
                    updated_at_unix_ms INTEGER NOT NULL,
                    FOREIGN KEY(source_key) REFERENCES metric_sources(source_key)
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
        if (
            batch.schema_identity.schema_id
            != self.catalog.schema_id_for_role(role)
        ):
            raise MetricEventContractError(
                "metric schema owner differs from durable source role"
            )
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
            self._changed.notify_all()
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
                SET status = 'committed', acknowledged_at_unix_ms = ?
                WHERE source_key = ? AND batch_sequence = ?
                """,
                (now_ms, key, str(int(batch.batch_sequence))),
            )
            self._connection.execute(
                """
                UPDATE metric_sources
                SET committed_batch_sequence = ?, committed_event_sequence = ?,
                    committed_digest = ?,
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
    ) -> training_pb2.MetricBatchCursor:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT committed_batch_sequence, committed_event_sequence,
                       committed_digest
                FROM metric_export_consumers WHERE source_key = ?
                """,
                (_source_key(source),),
            ).fetchone()
            if row is None:
                return training_pb2.MetricBatchCursor(source=source)
            cursor = training_pb2.MetricBatchCursor(
                source=source,
                acknowledged_batch_sequence=int(
                    row["committed_batch_sequence"]
                ),
                acknowledged_event_sequence=int(
                    row["committed_event_sequence"]
                ),
            )
            if row["committed_digest"]:
                cursor.acknowledged_batch_digest.CopyFrom(
                    _content_digest(str(row["committed_digest"]))
                )
            return cursor

    def next_export_batch(
        self,
        source: common_pb2.ServiceInstanceIdentity,
        cursor: training_pb2.MetricBatchCursor,
    ) -> training_pb2.MetricBatch | None:
        if not _same_message(cursor.source, source):
            raise MetricEventContractError(
                "metric export cursor source identity mismatch"
            )
        next_sequence = int(cursor.acknowledged_batch_sequence) + 1
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload, batch_digest FROM metric_batches
                WHERE source_key = ? AND batch_sequence = ?
                  AND status = 'committed'
                """,
                (_source_key(source), str(next_sequence)),
            ).fetchone()
        if row is None:
            return None
        batch = training_pb2.MetricBatch.FromString(row["payload"])
        if (
            int(batch.batch_sequence) != next_sequence
            or batch.batch_digest.hex != row["batch_digest"]
            or _digest_message(batch) != row["batch_digest"]
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
        cursor: training_pb2.MetricBatchCursor,
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
                    committed_event_sequence = ?, committed_digest = ?,
                    updated_at_unix_ms = ?
                WHERE source_key = ?
                """,
                (
                    str(int(cursor.acknowledged_batch_sequence)),
                    str(int(cursor.acknowledged_event_sequence)),
                    cursor.acknowledged_batch_digest.hex,
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
    training_pb2_grpc.MetricEventServiceServicer
):
    """Expose the Learner-owned raw journal to one exact Infra consumer."""

    def __init__(
        self,
        *,
        store: RawMetricBatchStore,
        contract: common_pb2.ContractIdentity,
        source: common_pb2.ServiceInstanceIdentity,
    ):
        self.store = store
        self.contract = _copy_message(contract)
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
        response = training_pb2.GetMetricBatchRsp()
        self._fill_availability(response)
        if (
            not _same_contract(request.contract, self.contract)
            or not self._valid_consumer(request.consumer)
            or not _same_message(request.cursor.source, self.source)
        ):
            response.result = (
                training_pb2.METRIC_BATCH_RESULT_REJECTED_IDENTITY
            )
            response.message = "metric consumer contract or identity is invalid"
            return response
        if not self.store.bind_export_consumer(
            self.source, request.consumer
        ):
            response.result = (
                training_pb2.METRIC_BATCH_RESULT_REJECTED_IDENTITY
            )
            response.message = "learner metric journal is pinned to another consumer"
            return response
        committed = self.store.export_cursor(self.source)
        if not _same_message(request.cursor, committed):
            response.result = training_pb2.METRIC_BATCH_RESULT_REJECTED_CURSOR
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
            response.result = training_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID
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
                        training_pb2.METRIC_BATCH_RESULT_REJECTED_INVALID
                    )
                    response.message = "requested limits are smaller than the next durable batch"
                    return response
                response.result = training_pb2.METRIC_BATCH_RESULT_DELIVERED
                response.message = "durable learner metric batch delivered"
                response.batch.CopyFrom(batch)
                self._fill_availability(response)
                return response
            _, _, source_final = self.store.export_availability(self.source)
            if source_final:
                response.result = training_pb2.METRIC_BATCH_RESULT_FINAL
                response.message = "learner metric source final batch is acknowledged"
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not context.is_active():
                response.result = training_pb2.METRIC_BATCH_RESULT_WAIT
                response.message = "no learner metric batch is currently available"
                return response
            self.store.wait_for_export_change(min(remaining, 0.25))

    def AckMetricBatch(self, request, context):
        del context
        response = training_pb2.AckMetricBatchRsp()
        self._fill_availability(response)
        if (
            not _same_contract(request.contract, self.contract)
            or not self._valid_consumer(request.consumer)
            or not _same_message(request.cursor.source, self.source)
            or not self.store.bind_export_consumer(
                self.source, request.consumer
            )
        ):
            response.result = (
                training_pb2.METRIC_BATCH_ACK_RESULT_REJECTED_IDENTITY
            )
            response.message = "metric ACK contract or consumer is invalid"
            response.committed_cursor.CopyFrom(
                self.store.export_cursor(self.source)
            )
            return response
        committed = self.store.export_cursor(self.source)
        response.committed_cursor.CopyFrom(committed)
        if _same_message(request.cursor, committed):
            response.result = (
                training_pb2.METRIC_BATCH_ACK_RESULT_ALREADY_APPLIED
            )
            response.message = "learner metric batch was already acknowledged"
            return response
        try:
            self.store.acknowledge_export(self.source, request.cursor)
        except MetricEventContractError as error:
            response.result = (
                training_pb2.METRIC_BATCH_ACK_RESULT_REJECTED_CURSOR
            )
            response.message = str(error)
            return response
        response.result = training_pb2.METRIC_BATCH_ACK_RESULT_APPLIED
        response.message = "learner metric batch acknowledged"
        response.committed_cursor.CopyFrom(
            self.store.export_cursor(self.source)
        )
        self._fill_availability(response)
        return response


def create_learner_metric_event_server(
    *,
    store: RawMetricBatchStore,
    contract: common_pb2.ContractIdentity,
    source: common_pb2.ServiceInstanceIdentity,
    port: int,
):
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise MetricEventContractError(
            "learner metric event server port must be in [1, 65535]"
        )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    training_pb2_grpc.add_MetricEventServiceServicer_to_server(
        LearnerMetricEventService(
            store=store,
            contract=contract,
            source=source,
        ),
        server,
    )
    bound = server.add_insecure_port(f"0.0.0.0:{port}")
    if bound != port:
        server.stop(0)
        raise MetricEventContractError(
            f"learner metric event server could not bind port {port}"
        )
    return server


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
        "minimum_behavior_model_step": None,
        "maximum_behavior_model_step": None,
        "behavior_model_lineages": set(),
    }


def _empty_train_statistics() -> dict:
    return {
        "train_update_count": 0,
        "actual_batch_size_sum": 0,
        "latest_train_update_sequence": 0,
        "latest_model_step": 0,
        "latest_cumulative_trained_samples": 0,
        "ppo": {},
        "minimum_behavior_model_step": None,
        "maximum_behavior_model_step": None,
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
    if source["minimum_behavior_model_step"] is not None:
        target["minimum_behavior_model_step"] = _merge_minimum(
            target["minimum_behavior_model_step"],
            source["minimum_behavior_model_step"],
        )
        target["maximum_behavior_model_step"] = _merge_maximum(
            target["maximum_behavior_model_step"],
            source["maximum_behavior_model_step"],
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
            "latest_model_step",
            "latest_cumulative_trained_samples",
        ):
            target[key] = source[key]
    for name, counts in source["ppo"].items():
        item = target["ppo"].setdefault(name, {"sum": 0.0, "count": 0})
        item["sum"] += counts["sum"]
        item["count"] += counts["count"]
    if source["minimum_behavior_model_step"] is not None:
        target["minimum_behavior_model_step"] = _merge_minimum(
            target["minimum_behavior_model_step"],
            source["minimum_behavior_model_step"],
        )
        target["maximum_behavior_model_step"] = _merge_maximum(
            target["maximum_behavior_model_step"],
            source["maximum_behavior_model_step"],
        )
    target["behavior_model_lineages"].update(
        source["behavior_model_lineages"]
    )


def _episode_event_statistics(
    fact: maze_metrics_pb2.EpisodeMetricFact,
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
        result["minimum_behavior_model_step"] = _merge_minimum(
            result["minimum_behavior_model_step"],
            int(agent.minimum_behavior_model_step),
        )
        result["maximum_behavior_model_step"] = _merge_maximum(
            result["maximum_behavior_model_step"],
            int(agent.maximum_behavior_model_step),
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


def _train_event_statistics(
    fact: training_metrics_pb2.TrainUpdateMetricFact,
) -> dict:
    result = _empty_train_statistics()
    result["train_update_count"] = 1
    result["actual_batch_size_sum"] = int(fact.actual_batch_size)
    result["latest_train_update_sequence"] = int(fact.train_update_sequence)
    result["latest_model_step"] = int(fact.published_model.model_step)
    result["latest_cumulative_trained_samples"] = int(
        fact.cumulative_trained_samples
    )
    result["minimum_behavior_model_step"] = int(
        fact.minimum_behavior_model_step
    )
    result["maximum_behavior_model_step"] = int(
        fact.maximum_behavior_model_step
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
            "latest_model_step": int(raw["latest_model_step"]),
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
        self._episode_first_observed_at: int | None = None
        self._episode_maximum_observed_at: int | None = None
        self._episode_last_observed_at: int | None = None
        self._episode_source_keys: set[str] = set()
        self._train_buckets: dict[int, dict] = {}
        self._train_all = _empty_train_statistics()
        self._train_latest: dict | None = None
        self._train_first_observed_at: int | None = None
        self._train_maximum_observed_at: int | None = None
        self._train_last_observed_at: int | None = None
        self._train_source_keys: set[str] = set()

    @staticmethod
    def _bucket_start(timestamp_unix_ms: int) -> int:
        return (int(timestamp_unix_ms) // LocalMetricProjector.BUCKET_MS) * (
            LocalMetricProjector.BUCKET_MS
        )

    def _accept_episode(
        self,
        source_key: str,
        event: training_pb2.MetricEvent,
        fact: maze_metrics_pb2.EpisodeMetricFact,
    ) -> None:
        statistics = _episode_event_statistics(fact)
        self._episode_source_keys.add(source_key)
        self._episode_recent.append(statistics)
        self._episode_latest = statistics
        _merge_episode_statistics(self._episode_all, statistics)
        observed_at = int(event.observed_at_unix_ms)
        self._episode_first_observed_at = (
            observed_at
            if self._episode_first_observed_at is None
            else min(self._episode_first_observed_at, observed_at)
        )
        self._episode_maximum_observed_at = (
            observed_at
            if self._episode_maximum_observed_at is None
            else max(self._episode_maximum_observed_at, observed_at)
        )
        self._episode_last_observed_at = observed_at
        bucket = self._episode_buckets.setdefault(
            self._bucket_start(observed_at),
            _empty_episode_statistics(),
        )
        _merge_episode_statistics(bucket, statistics)

    def _accept_train(
        self,
        source_key: str,
        event: training_pb2.MetricEvent,
        fact: training_metrics_pb2.TrainUpdateMetricFact,
    ) -> None:
        statistics = _train_event_statistics(fact)
        self._train_source_keys.add(source_key)
        self._train_latest = statistics
        _merge_train_statistics(self._train_all, statistics)
        observed_at = int(event.observed_at_unix_ms)
        self._train_first_observed_at = (
            observed_at
            if self._train_first_observed_at is None
            else min(self._train_first_observed_at, observed_at)
        )
        self._train_maximum_observed_at = (
            observed_at
            if self._train_maximum_observed_at is None
            else max(self._train_maximum_observed_at, observed_at)
        )
        self._train_last_observed_at = observed_at
        bucket = self._train_buckets.setdefault(
            self._bucket_start(observed_at),
            _empty_train_statistics(),
        )
        _merge_train_statistics(bucket, statistics)

    @staticmethod
    def _window_statistics(
        buckets: dict[int, dict],
        start_unix_ms: int,
        end_unix_ms: int,
        factory: Callable[[], dict],
        merge: Callable[[dict, dict], None],
    ) -> dict:
        result = factory()
        for bucket_start, statistics in buckets.items():
            if start_unix_ms <= bucket_start <= end_unix_ms:
                merge(result, statistics)
        return result

    def _advance(self) -> None:
        batches = self.store.committed_batches_after(self._last_row_id)
        for row_id, role, source_key, batch in batches:
            for event in batch.events:
                schema_id, fact = self.store.catalog.decode(
                    batch.schema_identity, event.fact_payload
                )
                if role == "aiserver" and schema_id == MAZE_METRIC_SCHEMA_ID:
                    self._accept_episode(source_key, event, fact)
                elif (
                    role == "learner"
                    and schema_id == TRAINING_METRIC_SCHEMA_ID
                ):
                    self._accept_train(source_key, event, fact)
                else:
                    raise MetricEventContractError(
                        "metric fact owner differs from durable source role"
                    )
            self._last_row_id = row_id
            self._view_revision += 1

    def _observed_window_bounds(
        self,
        first_observed_at: int | None,
        maximum_observed_at: int | None,
        duration_ms: int,
    ) -> tuple[int, int]:
        if first_observed_at is None or maximum_observed_at is None:
            return (0, 0)
        first_bucket = self._bucket_start(first_observed_at)
        nominal_start_bucket = self._bucket_start(
            maximum_observed_at - duration_ms
        )
        return (
            max(first_bucket, nominal_start_bucket),
            self._bucket_start(maximum_observed_at),
        )

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
                start_unix_ms, end_bucket_unix_ms = self._observed_window_bounds(
                    self._episode_first_observed_at,
                    self._episode_maximum_observed_at,
                    duration,
                )
                raw = self._window_statistics(
                    self._episode_buckets,
                    start_unix_ms,
                    end_bucket_unix_ms,
                    _empty_episode_statistics,
                    _merge_episode_statistics,
                )
                episode_windows[label] = _render_episode_statistics(
                    raw,
                    status=status,
                    window_kind="observed_time",
                    start_unix_ms=start_unix_ms,
                    end_unix_ms=(
                        self._episode_maximum_observed_at or 0
                    ),
                )
            episode_windows["all"] = _render_episode_statistics(
                self._episode_all,
                status=status,
                window_kind="run_to_date",
            )

            train_windows = {}
            for label, duration in self.TIME_WINDOWS_MS.items():
                start_unix_ms, end_bucket_unix_ms = self._observed_window_bounds(
                    self._train_first_observed_at,
                    self._train_maximum_observed_at,
                    duration,
                )
                raw = self._window_statistics(
                    self._train_buckets,
                    start_unix_ms,
                    end_bucket_unix_ms,
                    _empty_train_statistics,
                    _merge_train_statistics,
                )
                train_windows[label] = _render_train_statistics(
                    raw,
                    status=status,
                    window_kind="observed_time",
                    start_unix_ms=start_unix_ms,
                    end_unix_ms=(self._train_maximum_observed_at or 0),
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
                "schema_identities": self.store.catalog.identities_document(),
                "view_revision": self._view_revision,
                "status": status,
                "multi_server_aggregation_performed": False,
                "server_source_policy": "sequential_run_scoped_lifecycles",
                "server_source_count": len(self._episode_source_keys),
                "learner_source_count": len(self._train_source_keys),
                "episodes": {
                    "first_observed_at_unix_ms": self._episode_first_observed_at,
                    "maximum_observed_at_unix_ms": (
                        self._episode_maximum_observed_at
                    ),
                    "last_observed_at_unix_ms": self._episode_last_observed_at,
                    "latest": latest_episode,
                    "windows": episode_windows,
                },
                "train_updates": {
                    "first_observed_at_unix_ms": self._train_first_observed_at,
                    "maximum_observed_at_unix_ms": (
                        self._train_maximum_observed_at
                    ),
                    "last_observed_at_unix_ms": self._train_last_observed_at,
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
        batch = training_pb2.MetricBatch(
            contract=self.store.contract,
            schema_identity=self.store.catalog.schema_identity(
                TRAINING_METRIC_SCHEMA_ID
            ),
            source=self.source,
            batch_sequence=int(committed.acknowledged_batch_sequence) + 1,
            created_at_unix_ms=created_at_unix_ms,
            first_event_sequence=first_event_sequence,
            last_event_sequence=last_event_sequence,
            gap=training_pb2.MetricSequenceGap(
                first_unavailable_event_sequence=first_event_sequence,
                last_unavailable_event_sequence=last_event_sequence,
                oldest_available_event_sequence=last_event_sequence + 1,
                reason="learner_train_update_fact_unavailable",
            ),
        )
        batch.batch_digest.CopyFrom(_content_digest(_digest_message(batch)))
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
            event = training_pb2.MetricEvent(
                event_sequence=event_sequence,
                observed_at_unix_ms=observed_at,
                fact_payload=fact.SerializeToString(deterministic=True),
            )
            batch = training_pb2.MetricBatch(
                contract=self.store.contract,
                schema_identity=self.store.catalog.schema_identity(
                    TRAINING_METRIC_SCHEMA_ID
                ),
                source=self.source,
                batch_sequence=batch_sequence,
                created_at_unix_ms=int(time.time() * 1000),
                first_event_sequence=event_sequence,
                last_event_sequence=event_sequence,
                events=[event],
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
            finalized_at_unix_ms = int(time.time() * 1000)
            batch = training_pb2.MetricBatch(
                contract=self.store.contract,
                schema_identity=self.store.catalog.schema_identity(
                    TRAINING_METRIC_SCHEMA_ID
                ),
                source=self.source,
                batch_sequence=int(committed.acknowledged_batch_sequence) + 1,
                created_at_unix_ms=finalized_at_unix_ms,
                heartbeat=True,
                source_final=True,
                final_event_sequence=int(
                    committed.acknowledged_event_sequence
                ),
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
    INITIAL_RETRY_DELAY_SEC = 0.5
    MAX_RETRY_DELAY_SEC = 5.0

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
        self._state_lock = threading.Lock()
        self._transport_state = "starting"
        self._transport_failure_count = 0
        self._transport_unavailable_since = 0.0
        self._transport_last_error = ""
        self._ever_connected = False

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
                    "AIServer metric relay became unavailable; training "
                    "continues and reconnect runs in background: %s",
                    message,
                )
            else:
                self._transport_state = "waiting"
                self.logger.info(
                    "AIServer metric relay is waiting for AIServer metric "
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
                    "AIServer metric relay recovered after %.1fs and %d "
                    "retry attempt(s)",
                    elapsed,
                    failure_count,
                )
            elif prior_state == "waiting":
                self.logger.info(
                    "AIServer metric relay connected after waiting %.1fs "
                    "and %d retry attempt(s)",
                    elapsed,
                    failure_count,
                )
            else:
                self.logger.info("AIServer metric relay connected")
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

    def _discover_source(self) -> common_pb2.ServiceInstanceIdentity:
        status = self.status_stub.GetAIServerStatus(
            training_pb2.AIServerStatusReq(), timeout=1.5
        )
        if not _same_contract(status.contract, self.contract):
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
        retry_delay = self.INITIAL_RETRY_DELAY_SEC
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
                self._record_transport_connected()
                retry_delay = self.INITIAL_RETRY_DELAY_SEC
                if source_final:
                    self._stop.wait(1.0)
            except grpc.RpcError as error:
                self._record_transport_unavailable(error)
                self._stop.wait(retry_delay)
                retry_delay = min(
                    self.MAX_RETRY_DELAY_SEC, retry_delay * 2.0
                )
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
                retry_delay = min(
                    self.MAX_RETRY_DELAY_SEC, retry_delay * 2.0
                )
            except Exception as error:
                self.logger.error("AIServer metric relay failed: %s", error)
                self._stop.wait(retry_delay)
                retry_delay = min(
                    self.MAX_RETRY_DELAY_SEC, retry_delay * 2.0
                )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.GET_RPC_TIMEOUT_SEC + 1.0)
        with self._state_lock:
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
                    "failed to record incomplete AIServer metric source: %s",
                    error,
                )
