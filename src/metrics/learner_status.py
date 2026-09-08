"""Learner's public committed-state and catalog services. No Preview dependency."""
import time
from proto.training import training_pb2 as wire
from proto.training import training_pb2_grpc as rpc
from proto.metrics import catalog_pb2 as metric_catalog_pb2
from proto.metrics import catalog_pb2_grpc as metric_catalog_pb2_grpc
from proto.metrics import registry_pb2 as metric_registry_pb2

# Source definitions include only explicitly published numeric measurements.
STATUS_FIELDS = (
    ("learner.model_step", "Model Step", "model_step", "count", "training_depth"),
    ("learner.train_update.total", "Train Update", "train_updates", "count", "training_depth"),
    ("learner.trained_samples.total", "Trained Samples", "trained_samples", "count", "sample_flow"),
    ("learner.ppo.max_importance_ratio", "Max Importance Ratio", "max_importance_ratio", "ratio", "ppo_stability"),
    ("learner.value.explained_variance", "Explained Variance", "explained_variance", "ratio", "ppo_stability"),
)

class LearnerStatusService(rpc.LearnerStatusServiceServicer):
    def __init__(self, source, snapshot):
        self.source = source
        self.snapshot = snapshot

    def catalog_entries(self):
        return [metric_catalog_pb2.MetricCatalogEntry(
            definition=metric_registry_pb2.MetricDefinition(
                metric_id=name, display_name=label, category=category, unit=unit, scope="learner",
                value_type=metric_registry_pb2.METRIC_VALUE_TYPE_SCALAR if unit == "ratio" else metric_registry_pb2.METRIC_VALUE_TYPE_UNSIGNED,
                aggregation=metric_registry_pb2.METRIC_AGGREGATION_LATEST),
            status_method="rl.training.v1.LearnerStatusService/GetLearnerStatus", status_field=field,
        ) for name, label, field, unit, category in STATUS_FIELDS]

    def GetLearnerStatus(self, request, context):
        values = self.snapshot()
        response = wire.LearnerStatusRsp(learner=self.source, timestamp_unix_ms=int(time.time() * 1000))
        for field in response.DESCRIPTOR.fields:
            if field.name in ("learner", "timestamp_unix_ms", "model"):
                continue
            value = values.get("error" if field.name == "last_error" else field.name)
            if value is not None:
                setattr(response, field.name, value)
        identity = values.get("model_identity")
        if identity:
            for key, value in identity.items():
                if key in response.model.DESCRIPTOR.fields_by_name:
                    setattr(response.model, key, value)
        return response

class MetricCatalogService(metric_catalog_pb2_grpc.MetricCatalogServiceServicer):
    def __init__(self, writer, status):
        self.writer = writer
        self.status = status

    def GetMetricCatalog(self, request, context):
        result = self.writer.catalog()
        result.entries.extend(self.status.catalog_entries())
        return result
