"""
训练指标存储后端（抽象接口 + JSON Lines 文件实现）

模块化设计：采集层和查询层通过 MetricsBackend 接口与存储层交互，
具体存储实现不进入采集层和查询层的调用契约。
"""

import abc
import json
import os
import time
import threading
import glob
from typing import List, Optional


class MetricsBackend(abc.ABC):
    """指标存储后端抽象基类"""

    @abc.abstractmethod
    def write(self, record: dict):
        """写入一条指标记录"""
        ...

    @abc.abstractmethod
    def query(self, since_step: int = 0, limit: int = 0) -> List[dict]:
        """查询 train_step > since_step 的记录，limit=0 表示不限制"""
        ...

    @abc.abstractmethod
    def latest(self) -> Optional[dict]:
        """返回最新一条记录，无数据返回 None"""
        ...

    @abc.abstractmethod
    def summary(self) -> dict:
        """返回统计摘要"""
        ...

    @abc.abstractmethod
    def close(self):
        """关闭资源"""
        ...


class DisabledMetricsBackend(MetricsBackend):
    """Fail-open sink used when the optional metrics store is unavailable."""

    def __init__(self, reason: str):
        self.reason = str(reason)

    def write(self, record: dict):
        del record

    def query(self, since_step: int = 0, limit: int = 0) -> List[dict]:
        del since_step, limit
        return []

    def latest(self) -> Optional[dict]:
        return None

    def summary(self) -> dict:
        return {"enabled": False, "reason": self.reason}

    def close(self):
        return None


class JsonlBackend(MetricsBackend):
    """
    JSON Lines 文件存储后端

    每次训练生成一个 .jsonl 文件，每行一条 JSON 指标记录。
    支持增量读取（尾部追加写入 + 文件偏移缓存），适合长时间训练监控。
    """

    def __init__(self, metrics_dir: str):
        """
        Args:
            metrics_dir: 指标数据目录（自动创建）
        """
        self._metrics_dir = os.path.abspath(metrics_dir)
        os.makedirs(self._metrics_dir, exist_ok=True)

        # 当前写入文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(
            self._metrics_dir, f"metrics_{timestamp}.jsonl"
        )
        self._file = open(self._filepath, "a", encoding="utf-8")
        self._lock = threading.Lock()

        # 无界训练不能把每秒完整记录永久留在 Learner 内存中。查询按需读 JSONL；
        # 这里只保留常量空间的 latest 与摘要累加器。
        self._latest_record: Optional[dict] = None
        self._total_written = 0
        self._best_pass_rate: Optional[float] = None
        self._best_mean_reward: Optional[float] = None

        print(f"[MetricsBackend] 指标文件: {self._filepath}")

    def write(self, record: dict):
        """写入一条指标记录（线程安全）"""
        with self._lock:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            self._file.write(line + "\n")
            self._total_written += 1

            self._file.flush()

            self._latest_record = json.loads(line)
            pass_rate = float(record.get("pass_rate", 0.0))
            mean_reward = float(record.get("mean_episode_reward", 0.0))
            self._best_pass_rate = (
                pass_rate
                if self._best_pass_rate is None
                else max(self._best_pass_rate, pass_rate)
            )
            self._best_mean_reward = (
                mean_reward
                if self._best_mean_reward is None
                else max(self._best_mean_reward, mean_reward)
            )

    def query(self, since_step: int = 0, limit: int = 0) -> List[dict]:
        """查询 train_step > since_step 的记录"""
        with self._lock:
            self._file.flush()
            result = []
            with open(self._filepath, "r", encoding="utf-8") as source:
                for raw_line in source:
                    record = json.loads(raw_line)
                    if record.get("train_step", 0) <= since_step:
                        continue
                    result.append(record)
                    if limit > 0 and len(result) >= limit:
                        break
            return result

    def latest(self) -> Optional[dict]:
        """返回最新一条记录"""
        with self._lock:
            return self._latest_record

    def summary(self) -> dict:
        """返回统计摘要"""
        with self._lock:
            if self._latest_record is None:
                return {
                    "total_steps": 0,
                    "best_pass_rate": 0.0,
                    "best_mean_reward": 0.0,
                    "latest_loss": 0.0,
                    "file": os.path.basename(self._filepath),
                }

            return {
                "total_steps": self._total_written,
                "best_pass_rate": self._best_pass_rate,
                "best_mean_reward": self._best_mean_reward,
                "latest_loss": self._latest_record.get("total_loss", 0.0),
                "file": os.path.basename(self._filepath),
            }

    def close(self):
        """关闭文件句柄"""
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                print(f"[MetricsBackend] 指标文件已关闭，共写入 {self._total_written} 条记录")

    def get_filepath(self) -> str:
        """返回当前指标文件路径"""
        return self._filepath

    def get_metrics_dir(self) -> str:
        """返回指标数据目录"""
        return self._metrics_dir

    def record_count(self) -> int:
        """返回当前记录总数"""
        with self._lock:
            return self._total_written


def create_backend(backend_type: str, metrics_dir: str) -> MetricsBackend:
    """
    工厂函数：根据配置创建存储后端

    Args:
        backend_type: 后端类型（当前支持 "jsonl"）
        metrics_dir: 指标数据目录
    Returns:
        MetricsBackend 实例
    """
    if backend_type == "jsonl":
        return JsonlBackend(metrics_dir)
    else:
        raise ValueError(f"不支持的存储后端类型: {backend_type}，当前仅支持 'jsonl'")
