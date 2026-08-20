"""
PPO 训练器

包含 Actor-Critic 网络定义、PPO 训练循环和 ONNX 模型导出。
AIServer 负责按 pinned model segment 计算未归一化 GAE/Value Target；
Learner 只做 batch 级 advantage 归一化、PPO/optimizer 和模型发布。
"""

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper
from torch.distributions import Categorical

from src.log.logger import setup_logger


# ---- Actor-Critic 独立编码器网络 ----
class ActorCritic(nn.Module):
    """
    Actor-Critic 独立编码器架构

    Policy 分支: obs_dim → hidden → hidden → action_dim (logits)
    Value 分支:  obs_dim → hidden → hidden → 1
    两个分支使用完全独立的编码器，不共享权重。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ---- Policy 分支 ----
        self.policy_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # ---- Value 分支 ----
        self.value_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（用于推理和 ONNX 导出）

        Args:
            obs: 观测向量 [batch, obs_dim]
        Returns:
            action_logits: 动作 logits [batch, action_dim]
            value: 状态价值 [batch, 1]
        """
        p = self.policy_encoder(obs)
        action_logits = self.policy_head(p)

        v = self.value_encoder(obs)
        value = self.value_head(v)

        return action_logits, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作（用于 PPO 训练）

        Args:
            obs: 观测向量 [batch, obs_dim]
            actions: 动作 ID [batch]
        Returns:
            log_probs: 动作 log 概率 [batch]
            values: 状态价值 [batch]
            entropy: 策略熵 [batch]
        """
        # Policy 分支
        p = self.policy_encoder(obs)
        logits = self.policy_head(p)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Value 分支
        v = self.value_encoder(obs)
        values = self.value_head(v).squeeze(-1)

        return log_probs, values, entropy


class PPOTrainer:
    """
    PPO 训练器：管理模型、优化器、训练循环和 ONNX 导出

    生命周期：
        1. __init__() 构建网络 + 优化器
        2. train_on_batch() 消费 AIServer processed transitions
        3. export_onnx() 导出公开训练模型
    """

    MAX_MODEL_STEP = (1 << 64) - 1

    def __init__(self, config: dict):
        """
        Args:
            config: 完整配置字典（包含 model 和 training 两个子节点）
        """
        self._logger = setup_logger("PPOTrainer")

        # ---- 读取模型参数 ----
        model_cfg = config["model"]
        self._obs_dim = int(model_cfg["obs_dim"])
        self._action_dim = int(model_cfg["action_dim"])
        self._hidden_dim = int(model_cfg["hidden_dim"])

        # ---- 读取训练超参 ----
        train_cfg = config["training"]
        self._lr = float(train_cfg["learning_rate"])
        self._clip_epsilon = float(train_cfg["clip_epsilon"])
        self._value_clip_epsilon = float(train_cfg["value_clip_epsilon"])
        self._entropy_coef = float(train_cfg["entropy_coef"])
        self._value_coef = float(train_cfg["value_coef"])
        self._max_grad_norm = float(train_cfg["max_grad_norm"])
        self._n_epochs = int(train_cfg["n_epochs"])
        self._train_batch_size = int(train_cfg["train_batch_size"])
        self._mini_batch_size = int(train_cfg["mini_batch_size"])
        self._normalize_advantage = bool(train_cfg["normalize_advantage"])
        self._seed = int(train_cfg["seed"])
        if self._value_clip_epsilon <= 0.0:
            raise ValueError("value_clip_epsilon must be positive")

        # ---- 构建网络 + 优化器 ----
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self._device = torch.device(train_cfg["device"])
        self._model = ActorCritic(self._obs_dim, self._action_dim, self._hidden_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        # ---- 公开训练步长；每次成功 PPO Train Update 恰好加一 ----
        self._model_step = 0
        self._model_weights_inherited = False
        self._last_raw_metric_sum_counts: dict[str, dict[str, float | int]] = {}

        self._logger.info(
            "PPOTrainer 初始化完成: obs_dim=%d, action_dim=%d, hidden=%d, device=%s",
            self._obs_dim, self._action_dim, self._hidden_dim, self._device,
        )
        self._logger.info(
            "优化器参数: lr=%.9g, clip=%.9g, entropy=%.9g, value=%.9g, epochs=%d, train_batch=%d, mini_batch=%d",
            self._lr, self._clip_epsilon,
            self._entropy_coef, self._value_coef, self._n_epochs,
            self._train_batch_size, self._mini_batch_size,
        )

    # ---- PPO 训练 ----
    @staticmethod
    def _importance_ratio(
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.exp(new_log_probs - old_log_probs)

    @classmethod
    def _clipped_policy_loss(
        cls,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = cls._importance_ratio(new_log_probs, old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        ) * advantages
        return -torch.minimum(unclipped, clipped).mean(), ratio

    @staticmethod
    def _clipped_value_loss(
        new_values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float,
    ) -> torch.Tensor:
        clipped_values = old_values + torch.clamp(
            new_values - old_values,
            -clip_epsilon,
            clip_epsilon,
        )
        return torch.maximum(
            (new_values - returns).pow(2),
            (clipped_values - returns).pow(2),
        ).mean()

    def train_on_batch(
        self,
        samples: List[dict],
    ) -> Dict[str, object]:
        """
        对一个完整 processed-transition train batch 执行 PPO 训练

        Args:
            samples: AIServer 已填充 advantage/value_target 的样本
        Returns:
            训练统计字典
        """
        if len(samples) != self._train_batch_size:
            raise ValueError(
                "processed-transition batch size differs from the configured "
                f"train_batch_size: actual={len(samples)} "
                f"configured={self._train_batch_size}"
            )
        behavior_steps = [int(sample["behavior_model_step"]) for sample in samples]
        policy_lags = [
            self._model_step - step for step in behavior_steps
        ]
        if any(lag < 0 for lag in policy_lags):
            raise ValueError("behavior model step is from the future")
        policy_lag = max(policy_lags)
        if self._model_step >= self.MAX_MODEL_STEP:
            raise RuntimeError(
                "model step exhausted the uint64 publication space"
            )

        # ---- 1. 转换为 Tensor ----
        obs = torch.tensor([s["observation"] for s in samples], dtype=torch.float32, device=self._device)
        actions = torch.tensor([s["action"] for s in samples], dtype=torch.long, device=self._device)
        old_log_probs = torch.tensor([s["old_log_probability"] for s in samples], dtype=torch.float32, device=self._device)
        old_values = torch.tensor([s["old_value_prediction"] for s in samples], dtype=torch.float32, device=self._device)
        advantages = torch.tensor([s["advantage"] for s in samples], dtype=torch.float32, device=self._device)
        td_returns = torch.tensor([s["value_target"] for s in samples], dtype=torch.float32, device=self._device)
        if (
            not torch.isfinite(obs).all()
            or not torch.isfinite(old_log_probs).all()
            or not torch.isfinite(old_values).all()
            or not torch.isfinite(advantages).all()
            or not torch.isfinite(td_returns).all()
            or (actions < 0).any()
            or (actions >= self._action_dim).any()
        ):
            raise ValueError("PPO batch contains invalid tensors")

        # ---- 2. Advantage 标准化 ----
        raw_advantage_sum = float(advantages.sum().item())
        raw_advantage_count = int(advantages.numel())
        if self._normalize_advantage and len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            advantages = advantages - adv_mean
            if float(adv_std.item()) > 0.0:
                advantages = advantages / adv_std

        # ---- 3. 多轮 mini-batch PPO 训练 ----
        n_samples = len(samples)
        total_gradient_norm = 0.0
        maximum_importance_ratio = 0.0
        importance_ratio_sum = 0.0
        importance_ratio_square_sum = 0.0
        total_updates = 0
        raw_sample_evaluation_count = 0
        raw_policy_loss_sum = 0.0
        raw_value_loss_sum = 0.0
        raw_entropy_sum = 0.0
        raw_clip_fraction_sum = 0.0
        raw_approx_kl_sum = 0.0
        raw_total_loss_sum = 0.0
        value_pred_mean = 0.0
        return_target_mean = 0.0
        explained_variance: float | None = None

        self._model.train()
        model_snapshot = copy.deepcopy(self._model.state_dict())
        optimizer_snapshot = copy.deepcopy(self._optimizer.state_dict())
        torch_rng_snapshot = torch.get_rng_state().clone()
        numpy_rng_snapshot = np.random.get_state()
        cuda_rng_snapshot = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        try:
            for epoch in range(self._n_epochs):
                # 随机打乱索引
                indices = torch.randperm(n_samples, device=self._device)

                # 切分 mini-batch
                for start in range(0, n_samples, self._mini_batch_size):
                    end = min(start + self._mini_batch_size, n_samples)
                    mb_indices = indices[start:end]

                    mb_obs = obs[mb_indices]
                    mb_actions = actions[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]
                    mb_old_values = old_values[mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_td_returns = td_returns[mb_indices]

                    # 前向传播：获取新策略下的 log_prob、value、entropy
                    new_log_probs, new_values, entropy = self._model.evaluate_actions(mb_obs, mb_actions)

                    # ---- Policy Loss（PPO-Clip）----
                    policy_loss, ratio = self._clipped_policy_loss(
                        new_log_probs,
                        mb_old_log_probs,
                        mb_advantages,
                        self._clip_epsilon,
                    )

                    # ---- Value Loss（以行为策略 V(s) 为裁剪基准）----
                    value_loss = self._clipped_value_loss(
                        new_values,
                        mb_old_values,
                        mb_td_returns,
                        self._value_clip_epsilon,
                    )

                    # ---- Entropy Loss ----
                    entropy_loss = -entropy.mean()

                    # ---- Total Loss ----
                    total_loss = policy_loss + self._value_coef * value_loss + self._entropy_coef * entropy_loss
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError("PPO loss is non-finite")

                    # ---- 反向传播 + 梯度裁剪 + 优化器更新 ----
                    self._optimizer.zero_grad()
                    total_loss.backward()
                    if any(
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                        for parameter in self._model.parameters()
                    ):
                        raise FloatingPointError("PPO gradient is non-finite")
                    gradient_norm = nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._max_grad_norm
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError(
                            "PPO gradient norm is non-finite"
                        )
                    self._optimizer.step()
                    if any(
                        not torch.isfinite(parameter).all()
                        for parameter in self._model.parameters()
                    ):
                        raise FloatingPointError(
                            "PPO parameter update is non-finite"
                        )

                    # ---- 统计 ----
                    with torch.no_grad():
                        clip_fraction = ((ratio - 1.0).abs() > self._clip_epsilon).float().mean().item()
                        log_ratio = new_log_probs - mb_old_log_probs
                        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                        if not torch.isfinite(ratio).all():
                            raise FloatingPointError(
                                "PPO importance ratio is non-finite"
                            )
                        maximum_importance_ratio = max(
                            maximum_importance_ratio, ratio.max().item()
                        )

                    mini_batch_count = int(mb_indices.numel())
                    raw_sample_evaluation_count += mini_batch_count
                    raw_policy_loss_sum += policy_loss.item() * mini_batch_count
                    raw_value_loss_sum += value_loss.item() * mini_batch_count
                    raw_entropy_sum += entropy.mean().item() * mini_batch_count
                    raw_clip_fraction_sum += clip_fraction * mini_batch_count
                    raw_approx_kl_sum += approx_kl * mini_batch_count
                    raw_total_loss_sum += total_loss.item() * mini_batch_count
                    importance_ratio_sum += float(ratio.sum().item())
                    importance_ratio_square_sum += float(
                        ratio.square().sum().item()
                    )

                    total_gradient_norm += float(gradient_norm)
                    total_updates += 1

            # Use the committed model state for value diagnostics. These
            # metrics describe target fitting; the sign of V(s) alone does not.
            with torch.no_grad():
                _, committed_values = self._model(obs)
                committed_values = committed_values.squeeze(-1)
                if not torch.isfinite(committed_values).all():
                    raise FloatingPointError(
                        "PPO committed value prediction is non-finite"
                    )
                value_pred_mean = committed_values.mean().item()
                return_target_mean = td_returns.mean().item()
                target_variance = td_returns.var(unbiased=False)
                if target_variance.item() > 0.0:
                    residual_variance = (
                        td_returns - committed_values
                    ).var(unbiased=False)
                    explained_variance = (
                        1.0 - residual_variance / target_variance
                    ).item()
                if not all(
                    np.isfinite(value)
                    for value in (
                        value_pred_mean,
                        return_target_mean,
                    )
                ) or (
                    explained_variance is not None
                    and not np.isfinite(explained_variance)
                ):
                    raise FloatingPointError(
                        "PPO value diagnostics are non-finite"
                    )
        except Exception:
            self._model.load_state_dict(model_snapshot)
            self._optimizer.load_state_dict(optimizer_snapshot)
            torch.set_rng_state(torch_rng_snapshot)
            np.random.set_state(numpy_rng_snapshot)
            if cuda_rng_snapshot:
                torch.cuda.set_rng_state_all(cuda_rng_snapshot)
            raise

        # ---- 4. 汇总统计 ----
        if total_updates == 0 or raw_sample_evaluation_count == 0:
            raise RuntimeError("PPO update executed no optimizer work")

        self._model_step += 1

        self._last_raw_metric_sum_counts = {
            "approx_kl": {
                "sum": raw_approx_kl_sum,
                "count": raw_sample_evaluation_count,
            },
            "clip_fraction": {
                "sum": raw_clip_fraction_sum,
                "count": raw_sample_evaluation_count,
            },
            "entropy": {
                "sum": raw_entropy_sum,
                "count": raw_sample_evaluation_count,
            },
            "gradient_norm": {
                "sum": total_gradient_norm,
                "count": total_updates,
            },
            "policy_lag": {
                "sum": float(sum(policy_lags)),
                "count": n_samples,
            },
            "policy_loss": {
                "sum": raw_policy_loss_sum,
                "count": raw_sample_evaluation_count,
            },
            "return_target": {
                "sum": float(td_returns.sum().item()),
                "count": n_samples,
            },
            "total_loss": {
                "sum": raw_total_loss_sum,
                "count": raw_sample_evaluation_count,
            },
            "value_loss": {
                "sum": raw_value_loss_sum,
                "count": raw_sample_evaluation_count,
            },
            "value_prediction": {
                "sum": float(committed_values.sum().item()),
                "count": n_samples,
            },
        }
        if any(
            int(value["count"]) <= 0
            or not np.isfinite(float(value["sum"]))
            for value in self._last_raw_metric_sum_counts.values()
        ):
            raise FloatingPointError(
                "PPO raw metric sum/count is invalid"
            )

        if importance_ratio_square_sum <= 0.0 or not all(
            np.isfinite(value)
            for value in (
                importance_ratio_sum,
                importance_ratio_square_sum,
                maximum_importance_ratio,
            )
        ):
            raise FloatingPointError("PPO importance-ratio facts are invalid")
        importance_ratio_ess = (
            importance_ratio_sum * importance_ratio_sum
            / importance_ratio_square_sum
        )

        stats = {
            "policy_loss": raw_policy_loss_sum / raw_sample_evaluation_count,
            "value_loss": raw_value_loss_sum / raw_sample_evaluation_count,
            "total_loss": raw_total_loss_sum / raw_sample_evaluation_count,
            "entropy": raw_entropy_sum / raw_sample_evaluation_count,
            "clip_fraction": (
                raw_clip_fraction_sum / raw_sample_evaluation_count
            ),
            "approx_kl": raw_approx_kl_sum / raw_sample_evaluation_count,
            "gradient_norm": total_gradient_norm / total_updates,
            "max_importance_ratio": maximum_importance_ratio,
            "importance_ratio_sum": importance_ratio_sum,
            "importance_ratio_square_sum": importance_ratio_square_sum,
            "importance_ratio_ess": importance_ratio_ess,
            "raw_advantage_sum": raw_advantage_sum,
            "raw_advantage_count": raw_advantage_count,
            "normalized_advantage_sum": float(advantages.sum().item()),
            "normalized_advantage_count": int(advantages.numel()),
            "value_pred_mean": value_pred_mean,
            "return_target_mean": return_target_mean,
            "explained_variance": explained_variance,
            "learning_rate": self._lr,
            "policy_lag": policy_lag,
            "minimum_behavior_model_step": min(behavior_steps),
            "maximum_behavior_model_step": max(behavior_steps),
            "policy_lag_sum": int(sum(policy_lags)),
            "policy_lag_count": len(policy_lags),
            "mean_policy_lag": float(np.mean(policy_lags)),
            "sample_evaluation_count": raw_sample_evaluation_count,
            "optimizer_step_count": total_updates,
            "model_step": self._model_step,
        }

        self._logger.info(
            "训练更新 model_step=%d: policy_loss=%.4f, value_loss=%.4f, entropy=%.4f, clip=%.3f, samples=%d",
            self._model_step, stats["policy_loss"], stats["value_loss"],
            stats["entropy"], stats["clip_fraction"], n_samples,
        )

        return stats

    # ---- ONNX 模型导出 ----
    def export_onnx(self, export_path: str):
        """
        将当前模型导出为 ONNX 格式（兼容 PyTorch 2.6+ 新版导出器）

        Args:
            export_path: ONNX 文件输出路径
        """
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        was_training = bool(self._model.training)
        self._model.eval()
        dummy_input = torch.zeros(1, self._obs_dim, device=self._device)

        export_kwargs = dict(
            input_names=["observation"],
            output_names=["action_logits", "value"],
            dynamic_axes={
                "observation": {0: "batch"},
                "action_logits": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=11,
        )

        # PyTorch 2.6+ 默认走 dynamo 导出路径，简单网络使用 TorchScript 导出即可
        try:
            try:
                torch.onnx.export(
                    self._model,
                    dummy_input,
                    export_path,
                    dynamo=False,
                    **export_kwargs,
                )
            except TypeError:
                # PyTorch < 2.6 不支持 dynamo 参数
                torch.onnx.export(
                    self._model, dummy_input, export_path, **export_kwargs
                )
        finally:
            self._model.train(was_training)

        self._logger.info("ONNX 模型已导出: %s (step=%d)", export_path, self._model_step)

    # ---- PyTorch Checkpoint 保存/加载（断点续训）----
    def save_checkpoint(self, path: str, metadata: dict | None = None):
        """保存 PyTorch checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "model_step": self._model_step,
            "model_training": bool(self._model.training),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "metadata": metadata or {},
        }, path)
        self._logger.info("Checkpoint 已保存: %s", path)

    def load_checkpoint(self, path: str):
        """加载 PyTorch checkpoint"""
        if not os.path.isfile(path):
            self._logger.warning("Checkpoint 不存在: %s", path)
            return False
        try:
            checkpoint = torch.load(
                path, map_location=self._device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "model_version" in checkpoint:
            raise RuntimeError("legacy model_version checkpoint is not accepted")
        self._model_step = int(checkpoint.get("model_step", 0))
        if self._model_step < 0 or self._model_step > self.MAX_MODEL_STEP:
            raise RuntimeError("checkpoint model_step is invalid")
        self._model_weights_inherited = False
        self._model.train(bool(checkpoint.get("model_training", True)))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        self._logger.info("Checkpoint 已加载: %s (step=%d)", path, self._model_step)
        return True

    def load_onnx_weights(self, path: str) -> bool:
        """Load canonical SaveModel.onnx weights into a fresh training execution."""
        if (
            self._model_step != 0
            or self._optimizer.state
            or self._model_weights_inherited
        ):
            raise RuntimeError(
                "model weights can only be inherited by a fresh trainer"
            )
        if not os.path.isfile(path):
            self._logger.warning("ONNX 模型不存在: %s", path)
            return False
        try:
            document = onnx.load(path, load_external_data=False)
            onnx.checker.check_model(document)
        except Exception as error:
            raise RuntimeError("inherited ONNX model is invalid") from error
        if any(initializer.external_data for initializer in document.graph.initializer):
            raise RuntimeError("inherited ONNX model must be self-contained")
        if [item.name for item in document.graph.input] != ["observation"]:
            raise RuntimeError("inherited ONNX input contract does not match")
        if [item.name for item in document.graph.output] != [
            "action_logits",
            "value",
        ]:
            raise RuntimeError("inherited ONNX output contract does not match")

        initializers = {
            initializer.name: initializer
            for initializer in document.graph.initializer
        }
        target_state = self._model.state_dict()
        if set(initializers) != set(target_state):
            raise RuntimeError("inherited ONNX parameter names do not match")
        source_state = {}
        for name, target in target_state.items():
            try:
                array = np.array(
                    numpy_helper.to_array(initializers[name]), copy=True
                )
            except Exception as error:
                raise RuntimeError(
                    f"inherited ONNX parameter is unreadable: {name}"
                ) from error
            source = torch.from_numpy(array)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise RuntimeError(
                    f"inherited ONNX parameter is incompatible: {name}"
                )
            if not torch.isfinite(source).all():
                raise RuntimeError(
                    f"inherited ONNX parameter is non-finite: {name}"
                )
            source_state[name] = source.to(self._device)
        self._model.load_state_dict(source_state, strict=True)
        self._model_weights_inherited = True
        self._logger.info(
            "ONNX 模型权重已继承: %s (fresh training step=0)", path
        )
        return True

    # ---- 属性访问 ----
    @property
    def model(self) -> ActorCritic:
        """返回 ActorCritic 模型实例"""
        return self._model

    @property
    def model_step(self) -> int:
        """返回当前公开训练模型步长。"""
        return self._model_step

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def raw_metric_sum_counts(self) -> dict[str, dict[str, float | int]]:
        """Return raw mergeable statistics from the last committed update."""
        return copy.deepcopy(self._last_raw_metric_sum_counts)
