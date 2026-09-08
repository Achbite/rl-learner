"""Query-owned arithmetic over complete producer intervals (no wire or training dependency)."""
def current_statistic(source, metric_id, at_unix_ms):
    bucket_ms = source["time_bucket_ms"]
    end = int(at_unix_ms) // bucket_ms * bucket_ms
    start = end - source["mean_window_ms"]
    definition = source["catalog"].get(metric_id)
    if definition is None: return None
    operation = definition["aggregation"]
    items = [interval["values"][metric_id]
             for interval in source.get("current_intervals", [])
             if start < interval["end_unix_ms"] <= end and metric_id in interval["values"]]
    if not items:
        return None
    statistic = {
        "window_start_unix_ms": start,
        "window_end_unix_ms": end,
        "interval_start_unix_ms": min(item["interval_start_unix_ms"] for item in items),
        "interval_end_unix_ms": max(item["interval_end_unix_ms"] for item in items),
    }
    if operation == "mean":
        count = sum(item["count"] for item in items)
        total = sum(item["sum"] for item in items)
        statistic.update(value=total / count, sum=total, count=count)
    else:
        values = [item["value"] for item in items]
        reducers = {"sum": sum, "min": min, "max": max}
        statistic["value"] = reducers[operation](values) if operation in reducers else values[-1]
    return statistic
