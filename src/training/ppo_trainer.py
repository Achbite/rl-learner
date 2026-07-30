"""
PPO 训练器

包含 Actor-Critic 网络定义、GAE 计算、PPO 训练循环、ONNX 模型导出。
训练架构：AIServer 负责特征提取 + 推理 + 奖励计算 + 样本打包，
         Learner 负责 GAE 计算 + PPO 训练 + ONNX 导出。
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.log.logger import setup_logger


# ---- Actor-Critic 独立编码器网络 ----
class ActorCritic(nn.Module):
    """
    Actor-Critic 独立编码器架构

    Policy 分支: obs_dim → hidden → hidden → action_dim (Softmax)
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
            action_probs: 动作概率 [batch, action_dim]
            value: 状态价值 [batch, 1]
        """
        p = self.policy_encoder(obs)
        action_probs = torch.softmax(self.policy_head(p), dim=-1)

        v = self.value_encoder(obs)
        value = self.value_head(v)

        return action_probs, value

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

    def __init__(self, config: dict):
        """
        Args:
            config: 完整配置字典（包含 model 和 training 两个子节点）
        """
        self._logger = setup_logger("PPOTrainer")

        # ---- 读取模型参数 ----
        model_cfg = config.get("model", {})
        self._obs_dim = model_cfg.get("obs_dim", 5)
        self._action_dim = model_cfg.get("action_dim", 9)
        self._hidden_dim = model_cfg.get("hidden_dim", 64)

        # ---- 读取训练超参 ----
        train_cfg = config.get("training", {})
        self._lr = train_cfg.get("learning_rate", 3e-4)
        self._gamma = train_cfg.get("gamma", 0.99)
        self._gae_lambda = train_cfg.get("gae_lambda", 0.95)
        self._clip_epsilon = train_cfg.get("clip_epsilon", 0.2)
        self._entropy_coef = train_cfg.get("entropy_coef", 0.01)
        self._value_coef = train_cfg.get("value_coef", 0.5)
        self._max_grad_norm = train_cfg.get("max_grad_norm", 0.5)
        self._n_epochs = train_cfg.get("n_epochs", 4)
        self._mini_batch_size = train_cfg.get("mini_batch_size", 64)
        self._normalize_advantage = train_cfg.get("normalize_advantage", True)
        self._seed = int(train_cfg.get("seed", 0))

        # ---- 构建网络 + 优化器 ----
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self._device = torch.device(train_cfg.get("device", "cpu"))
        self._model = ActorCritic(self._obs_dim, self._action_dim, self._hidden_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        # ---- 模型版本号 ----
        self._model_version = 0

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
        if not bootstrap_valid or not np.isfinite(bootstrap_value):
            raise ValueError("fragment bootstrap is missing or non-finite")

        values = np.asarray(
            [sample["old_vpred"] for sample in trajectory],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("fragment old_vpred contains non-finite values")

        # ---- 1. 逆序计算 GAE ----
        rewards = np.array([s["reward"] for s in trajectory], dtype=np.float32)
        if not np.all(np.isfinite(rewards)):
            raise ValueError("fragment reward contains non-finite values")
        T = len(trajectory)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = float(bootstrap_value)
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
    def train_on_batch(self, samples: List[dict]) -> Dict[str, float]:
        """
        对一批已计算好 GAE 的样本执行 PPO 训练

        Args:
            samples: 样本 dict 列表（已填充 advantage 和 td_return）
        Returns:
            训练统计字典
        """
        if not samples:
            return self._empty_stats()

        # ---- 1. 转换为 Tensor ----
        obs = torch.tensor([s["obs"] for s in samples], dtype=torch.float32, device=self._device)
        actions = torch.tensor([s["action"] for s in samples], dtype=torch.long, device=self._device)
        old_log_probs = torch.tensor([s["old_log_prob"] for s in samples], dtype=torch.float32, device=self._device)
        advantages = torch.tensor([s["advantage"] for s in samples], dtype=torch.float32, device=self._device)
        td_returns = torch.tensor([s["td_return"] for s in samples], dtype=torch.float32, device=self._device)

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
        total_updates = 0

        self._model.train()

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
                mb_advantages = advantages[mb_indices]
                mb_td_returns = td_returns[mb_indices]

                # 前向传播：获取新策略下的 log_prob、value、entropy
                new_log_probs, new_values, entropy = self._model.evaluate_actions(mb_obs, mb_actions)

                # ---- Policy Loss（PPO-Clip）----
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value Loss ----
                value_loss = F.mse_loss(new_values, mb_td_returns)

                # ---- Entropy Loss ----
                entropy_loss = -entropy.mean()

                # ---- Total Loss ----
                total_loss = policy_loss + self._value_coef * value_loss + self._entropy_coef * entropy_loss

                # ---- 反向传播 + 梯度裁剪 + 优化器更新 ----
                self._optimizer.zero_grad()
                total_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._max_grad_norm
                )
                self._optimizer.step()

                # ---- 统计 ----
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self._clip_epsilon).float().mean().item()
                    log_ratio = new_log_probs - mb_old_log_probs
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_fraction += clip_fraction
                total_approx_kl += approx_kl
                total_gradient_norm += float(gradient_norm)
                total_combined_loss += total_loss.item()
                total_updates += 1

        # ---- 4. 汇总统计 ----
        if total_updates == 0:
            return self._empty_stats()

        self._model_version += 1

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
            "mean_advantage": round(advantages.mean().item(), 6),
            "learning_rate": self._lr,
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

        self._model.eval()
        dummy_input = torch.zeros(1, self._obs_dim, device=self._device)

        export_kwargs = dict(
            input_names=["obs"],
            output_names=["action_probs", "value"],
            dynamic_axes={
                "obs": {0: "batch"},
                "action_probs": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=11,
        )

        # PyTorch 2.6+ 默认走 dynamo 导出路径，简单网络使用 TorchScript 导出即可
        try:
            torch.onnx.export(self._model, dummy_input, export_path, dynamo=False, **export_kwargs)
        except TypeError:
            # PyTorch < 2.6 不支持 dynamo 参数
            torch.onnx.export(self._model, dummy_input, export_path, **export_kwargs)

        self._logger.info("ONNX 模型已导出: %s (version=%d)", export_path, self._model_version)

    # ---- PyTorch Checkpoint 保存/加载（断点续训）----
    def save_checkpoint(self, path: str, metadata: dict | None = None):
        """保存 PyTorch checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "model_version": self._model_version,
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
        self._model_version = checkpoint.get("model_version", 0)
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        self._logger.info("Checkpoint 已加载: %s (version=%d)", path, self._model_version)
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

    # ---- 内部工具 ----
    def _empty_stats(self) -> Dict[str, float]:
        """返回空训练统计（无样本时使用）"""
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "gradient_norm": 0.0,
            "mean_advantage": 0.0,
            "learning_rate": self._lr,
            "model_version": self._model_version,
        }
