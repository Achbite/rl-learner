"""
PPO 训练器

包含 Actor-Critic 网络定义、GAE 计算、PPO 训练循环、ONNX 模型导出。
训练架构：AIServer 负责特征提取 + 推理 + 奖励计算 + 样本打包，
         Learner 负责 GAE 计算 + PPO 训练 + ONNX 导出。
"""

import copy
import os
from collections.abc import Mapping
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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
    PPO 训练器：管理模型、优化器、GAE 计算、训练循环、ONNX 导出

    生命周期：
        1. __init__() 构建网络 + 优化器
        2. compute_gae() 对每条 trajectory 计算 GAE
        3. train_on_batch() 执行 PPO 训练
        4. export_onnx() 定期导出模型
    """

    MAX_MODEL_VERSION = (1 << 64) - 1

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
        self._gamma = float(train_cfg["gamma"])
        self._gae_lambda = float(train_cfg["gae_lambda"])
        self._clip_epsilon = float(train_cfg["clip_epsilon"])
        self._value_clip_epsilon = float(train_cfg["value_clip_epsilon"])
        self._entropy_coef = float(train_cfg["entropy_coef"])
        self._value_coef = float(train_cfg["value_coef"])
        self._max_grad_norm = float(train_cfg["max_grad_norm"])
        self._n_epochs = int(train_cfg["n_epochs"])
        self._mini_batch_size = int(train_cfg["mini_batch_size"])
        self._normalize_advantage = bool(train_cfg["normalize_advantage"])
        self._max_policy_lag = int(train_cfg["max_policy_lag"])
        self._seed = int(train_cfg["seed"])
        if self._value_clip_epsilon <= 0.0:
            raise ValueError("value_clip_epsilon must be positive")
        if self._max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")

        # ---- 构建网络 + 优化器 ----
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self._device = torch.device(train_cfg["device"])
        self._model = ActorCritic(self._obs_dim, self._action_dim, self._hidden_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        # ---- 模型版本号 ----
        self._model_version = 0
        self._model_weights_inherited = False
        self._last_raw_metric_sum_counts: dict[str, dict[str, float | int]] = {}

        self._logger.info(
            "PPOTrainer 初始化完成: obs_dim=%d, action_dim=%d, hidden=%d, device=%s",
            self._obs_dim, self._action_dim, self._hidden_dim, self._device,
        )
        self._logger.info(
            "超参: lr=%.4f, gamma=%.2f, lambda=%.2f, clip=%.2f, entropy=%.3f, value=%.2f, epochs=%d, mini_batch=%d",
            self._lr, self._gamma, self._gae_lambda, self._clip_epsilon,
            self._entropy_coef, self._value_coef, self._n_epochs, self._mini_batch_size,
        )

    # ---- GAE 计算 ----
    def compute_gae(
        self,
        trajectory: List[dict],
        bootstrap_value: float,
        bootstrap_valid: bool,
    ) -> List[dict]:
        """
        对一条 trajectory 计算 GAE 优势值和 TD(λ) 回报

        Args:
            trajectory: 样本 dict 列表（按时间序列排列）
            bootstrap_value: fragment 末端 V(s_{t+1})，由行为模型采样时计算
            bootstrap_valid: bootstrap 是否经过 AIServer 校验
        Returns:
            填充了 advantage 和 td_return 的 trajectory
        """
        if not trajectory:
            return trajectory
        for index, sample in enumerate(trajectory):
            terminated = bool(sample.get("terminated", False))
            truncated = bool(sample.get("truncated", False))
            if terminated and truncated:
                raise ValueError("transition cannot be terminated and truncated")
            if index != len(trajectory) - 1 and (terminated or truncated):
                raise ValueError("trajectory continues after an end transition")

        terminal_end = bool(trajectory[-1].get("terminated", False))
        if terminal_end:
            if bootstrap_valid or float(bootstrap_value) != 0.0:
                raise ValueError("terminated fragment must not bootstrap")
        elif not bootstrap_valid or not np.isfinite(bootstrap_value):
            raise ValueError("continuing or truncated fragment requires bootstrap")

        values = np.asarray(
            [sample["old_value_prediction"] for sample in trajectory],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "fragment old_value_prediction contains non-finite values"
            )

        # ---- 1. 逆序计算 GAE ----
        rewards = np.array([s["reward"] for s in trajectory], dtype=np.float32)
        if not np.all(np.isfinite(rewards)):
            raise ValueError("fragment reward contains non-finite values")
        T = len(trajectory)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = 0.0 if terminal_end else float(bootstrap_value)
            else:
                next_val = values[t + 1]

            nonterminal = 0.0 if trajectory[t].get("terminated", False) else 1.0
            # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self._gamma * next_val * nonterminal - values[t]
            # Â_t = δ_t + γλ * Â_{t+1}
            gae = (
                delta
                + self._gamma
                * self._gae_lambda
                * nonterminal
                * gae
            )
            advantages[t] = gae

        # td_return = Â_t + V(s_t)
        td_returns = advantages + values

        # ---- 2. 填充到样本中 ----
        for i, sample in enumerate(trajectory):
            sample["advantage"] = float(advantages[i])
            sample["td_return"] = float(td_returns[i])

        return trajectory

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
        behavior_model_version: int | None = None,
    ) -> Dict[str, float]:
        """
        对一批已计算好 GAE 的样本执行 PPO 训练

        Args:
            samples: 样本 dict 列表（已填充 advantage 和 td_return）
        Returns:
            训练统计字典
        """
        behavior_versions = [
            int(
                sample.get(
                    "behavior_model_version",
                    self._model_version
                    if behavior_model_version is None
                    else behavior_model_version,
                )
            )
            for sample in samples
        ]
        policy_lags = [
            self._model_version - version for version in behavior_versions
        ]
        if any(lag < 0 for lag in policy_lags):
            raise ValueError("behavior model version is from the future")
        if any(lag > self._max_policy_lag for lag in policy_lags):
            raise ValueError(
                "behavior model version exceeds max_policy_lag: "
                f"current={self._model_version} "
                f"behavior=[{min(behavior_versions)},"
                f"{max(behavior_versions)}] "
                f"max={self._max_policy_lag}"
            )
        policy_lag = max(policy_lags, default=0)
        if not samples:
            self._last_raw_metric_sum_counts = {}
            return self._empty_stats(policy_lag)
        if self._model_version >= self.MAX_MODEL_VERSION:
            raise RuntimeError(
                "model version exhausted the uint64 publication space"
            )

        # ---- 1. 转换为 Tensor ----
        obs = torch.tensor([s["observation"] for s in samples], dtype=torch.float32, device=self._device)
        actions = torch.tensor([s["action"] for s in samples], dtype=torch.long, device=self._device)
        old_log_probs = torch.tensor([s["old_log_probability"] for s in samples], dtype=torch.float32, device=self._device)
        old_values = torch.tensor([s["old_value_prediction"] for s in samples], dtype=torch.float32, device=self._device)
        advantages = torch.tensor([s["advantage"] for s in samples], dtype=torch.float32, device=self._device)
        td_returns = torch.tensor([s["td_return"] for s in samples], dtype=torch.float32, device=self._device)
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
        if self._normalize_advantage and len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # ---- 3. 多轮 mini-batch PPO 训练 ----
        n_samples = len(samples)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_gradient_norm = 0.0
        total_combined_loss = 0.0
        maximum_importance_ratio = 0.0
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
        explained_variance = 0.0

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

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    total_clip_fraction += clip_fraction
                    total_approx_kl += approx_kl
                    total_gradient_norm += float(gradient_norm)
                    total_combined_loss += total_loss.item()
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
                if target_variance.item() > 1e-12:
                    residual_variance = (
                        td_returns - committed_values
                    ).var(unbiased=False)
                    explained_variance = (
                        1.0 - residual_variance / target_variance
                    ).item()
                else:
                    explained_variance = 0.0
                if not all(
                    np.isfinite(value)
                    for value in (
                        value_pred_mean,
                        return_target_mean,
                        explained_variance,
                    )
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
        if total_updates == 0:
            return self._empty_stats(policy_lag)

        self._model_version += 1

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
            self._logger.error(
                "PPO raw metric sum/count is invalid; metric fact disabled for "
                "this committed update"
            )
            self._last_raw_metric_sum_counts = {}

        stats = {
            "policy_loss": round(total_policy_loss / total_updates, 6),
            "value_loss": round(total_value_loss / total_updates, 6),
            "total_loss": round(total_combined_loss / total_updates, 6),
            "entropy": round(total_entropy / total_updates, 6),
            "clip_fraction": round(total_clip_fraction / total_updates, 4),
            "approx_kl": round(total_approx_kl / total_updates, 8),
            "gradient_norm": round(
                total_gradient_norm / total_updates, 6
            ),
            "max_importance_ratio": round(maximum_importance_ratio, 6),
            "mean_advantage": round(advantages.mean().item(), 6),
            "value_pred_mean": round(value_pred_mean, 6),
            "return_target_mean": round(return_target_mean, 6),
            "explained_variance": round(explained_variance, 6),
            "learning_rate": self._lr,
            "policy_lag": policy_lag,
            "minimum_behavior_model_version": min(behavior_versions),
            "maximum_behavior_model_version": max(behavior_versions),
            "mean_policy_lag": round(float(np.mean(policy_lags)), 6),
            "max_policy_lag": self._max_policy_lag,
            "model_version": self._model_version,
        }

        self._logger.info(
            "训练步骤 v%d: policy_loss=%.4f, value_loss=%.4f, entropy=%.4f, clip=%.3f, samples=%d",
            self._model_version, stats["policy_loss"], stats["value_loss"],
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

        self._logger.info("ONNX 模型已导出: %s (version=%d)", export_path, self._model_version)

    # ---- PyTorch Checkpoint 保存/加载（断点续训）----
    def save_checkpoint(self, path: str, metadata: dict | None = None):
        """保存 PyTorch checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "model_version": self._model_version,
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
        self._model_version = int(checkpoint.get("model_version", 0))
        self._model_weights_inherited = False
        self._model.train(bool(checkpoint.get("model_training", True)))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        self._logger.info("Checkpoint 已加载: %s (version=%d)", path, self._model_version)
        return True

    def load_model_weights(self, path: str) -> bool:
        """Load only model parameters into a fresh training lineage."""
        if (
            self._model_version != 0
            or self._optimizer.state
            or self._model_weights_inherited
        ):
            raise RuntimeError(
                "model weights can only be inherited by a fresh trainer"
            )
        if not os.path.isfile(path):
            self._logger.warning("Checkpoint 不存在: %s", path)
            return False
        try:
            checkpoint = torch.load(
                path, map_location=self._device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self._device)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("inherited checkpoint is not a mapping")
        source_state = checkpoint.get("model_state_dict")
        if not isinstance(source_state, Mapping):
            raise RuntimeError("inherited checkpoint has no model state")
        target_state = self._model.state_dict()
        if set(source_state) != set(target_state):
            raise RuntimeError("inherited model parameter names do not match")
        for name, target in target_state.items():
            source = source_state[name]
            if (
                not isinstance(source, torch.Tensor)
                or source.shape != target.shape
                or source.dtype != target.dtype
            ):
                raise RuntimeError(
                    f"inherited model parameter is incompatible: {name}"
                )
        self._model.load_state_dict(source_state, strict=True)
        self._model_weights_inherited = True
        # The optimizer, RNG, counters and publication identity intentionally
        # remain those of this newly constructed trainer.
        self._logger.info(
            "Checkpoint 模型权重已继承: %s (new lineage version=0)", path
        )
        return True

    # ---- 属性访问 ----
    @property
    def model(self) -> ActorCritic:
        """返回 ActorCritic 模型实例"""
        return self._model

    @property
    def model_version(self) -> int:
        """返回当前模型版本号"""
        return self._model_version

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def max_policy_lag(self) -> int:
        return self._max_policy_lag

    def raw_metric_sum_counts(self) -> dict[str, dict[str, float | int]]:
        """Return raw mergeable statistics from the last committed update."""
        return copy.deepcopy(self._last_raw_metric_sum_counts)

    # ---- 内部工具 ----
    def _empty_stats(self, policy_lag: int = 0) -> Dict[str, float]:
        """返回空训练统计（无样本时使用）"""
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "gradient_norm": 0.0,
            "max_importance_ratio": 0.0,
            "mean_advantage": 0.0,
            "value_pred_mean": 0.0,
            "return_target_mean": 0.0,
            "explained_variance": 0.0,
            "learning_rate": self._lr,
            "policy_lag": policy_lag,
            "max_policy_lag": self._max_policy_lag,
            "model_version": self._model_version,
        }
