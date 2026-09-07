"""Metric definitions owned by the PPO/update producer, not by the reader."""

from proto import training_pb2 as wire
from .registered_metrics import MetricRegistry


class TrainMetricProducer:
    def __init__(self):
        self.registry = MetricRegistry()
        # Match the actual loops in PPOTrainer: evaluations include PPO epochs;
        # optimizer steps and batch transitions have distinct denominators.
        for name, denominator, unit in (
            ("raw_advantage", "transition", "advantage"),
            ("normalized_advantage", "transition", "normalized_advantage"),
            ("approx_kl", "sample_evaluation", "nats"),
            ("clip_fraction", "sample_evaluation", "ratio"),
            ("entropy", "sample_evaluation", "nats"),
            ("gradient_norm", "optimizer_step", "norm"),
            ("policy_lag", "transition", "model_step"),
            ("policy_loss", "sample_evaluation", "loss"),
            ("return_target", "transition", "reward"),
            ("total_loss", "sample_evaluation", "loss"),
            ("value_loss", "sample_evaluation", "loss"),
            ("value_prediction", "transition", "reward"),
        ):
            self.registry.register(
                "learner.ppo." + name,
                display_name="Approx. KL" if name == "approx_kl" else name.replace("_", " ").title(),
                unit=unit,
                scope="train_update", denominator=denominator,
                value_type=wire.METRIC_VALUE_TYPE_SUM_COUNT,
                aggregation=wire.METRIC_AGGREGATION_MEAN,
            )
        self.counters = (
            "train_update_sequence", "cumulative_trained_samples", "actual_batch_size",
            "minimum_behavior_model_step", "maximum_behavior_model_step",
            "requested_train_batch_size", "pool_draw_slot_count", "unique_item_count",
            "duplicate_item_slot_count", "sample_evaluation_count", "optimizer_step_count",
        )
        for name in (*self.counters, "model_step"):
            self.registry.register(
                "learner.update." + name, display_name=name.replace("_", " ").title(), unit="count",
                scope="train_update", value_type=wire.METRIC_VALUE_TYPE_UNSIGNED,
                aggregation=wire.METRIC_AGGREGATION_LATEST,
            )

    def record(self, fact):
        points = [wire.MetricPoint(
            metric_id="learner.ppo." + item.field_id,
            sum_count=wire.MetricSumCount(sum=item.sum, count=item.count),
        ) for item in fact.ppo_statistics]
        points.extend(wire.MetricPoint(
            metric_id="learner.update." + name, unsigned_value=getattr(fact, name),
        ) for name in self.counters)
        points.append(wire.MetricPoint(
            metric_id="learner.update.model_step",
            unsigned_value=fact.published_model.model_step,
        ))
        return self.registry.record(points, attributes={
            "train_update_id": fact.train_update_id, "delivery_id": fact.delivery_id,
            "model_lineage_id": fact.published_model.model_lineage_id,
            "behavior_model_lineage_id": fact.behavior_model_lineage_id,
        })
