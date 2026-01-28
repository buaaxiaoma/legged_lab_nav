import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedRewardModel(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


class RewardLearningManager:
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.device = env.device
        self.num_envs = env.num_envs

        if not self.enabled:
            self.model = None
            return

        if "critic" not in env.observation_manager.group_obs_dim:
            self.enabled = False
            self.model = None
            return

        weight_init = getattr(cfg, "weight_init", {}) or {}
        self.term_names = [name for name in env.reward_manager.active_terms if name != "learned_reward"]
        self.term_cfgs = [env.reward_manager.get_term_cfg(name) for name in self.term_names]
        self.weights = torch.tensor([weight_init.get(name, term.weight) for name, term in zip(self.term_names, self.term_cfgs)],
                                    device=self.device)
        self.initial_weights = self.weights.clone()
        for name, weight in zip(self.term_names, self.weights):
            env.reward_manager.get_term_cfg(name).weight = float(weight.item())

        self.target_terms = list(getattr(cfg, "target_terms", []))
        if not self.target_terms:
            self.target_terms = list(self.term_names)
        self.target_indices = [self.term_names.index(name) for name in self.target_terms if name in self.term_names]

        self._episode_term_sums = torch.zeros((self.num_envs, len(self.term_names)), device=self.device)
        self._episode_term_history: list[torch.Tensor] = []
        self._episode_target_history: list[torch.Tensor] = []
        self._last_learned_reward_loss: float | None = None

        obs_dim = env.observation_manager.group_obs_dim["critic"][0]
        self.model = LearnedRewardModel(obs_dim, list(cfg.learned_reward_hidden_dims)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learned_reward_lr)
        self._obs_buffer = torch.zeros((cfg.learned_reward_buffer_size, obs_dim), device=self.device)
        self._target_buffer = torch.zeros((cfg.learned_reward_buffer_size, 1), device=self.device)
        self._buffer_pos = 0
        self._buffer_full = False
        self._last_critic_obs: torch.Tensor | None = None

    def set_last_critic_obs(self, obs: torch.Tensor | None):
        if not self.enabled:
            return
        if obs is None:
            self._last_critic_obs = None
            return
        self._last_critic_obs = obs.detach()

    def get_reward(self) -> torch.Tensor:
        if not self.enabled or self.model is None or self._last_critic_obs is None:
            return torch.zeros(self.num_envs, device=self.device)
        with torch.no_grad():
            reward = self.model(self._last_critic_obs).squeeze(-1)
        return reward

    def record_step(self, env, critic_obs: torch.Tensor | None):
        if not self.enabled:
            return
        raw_terms = self._compute_raw_terms(env)
        self._episode_term_sums += raw_terms
        if critic_obs is None:
            return
        targets = self._extract_target(raw_terms).unsqueeze(-1)
        self._store_obs_target(critic_obs.detach(), targets.detach())

    def on_reset(self, env_ids):
        if not self.enabled:
            return
        if env_ids is None or len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device)
        term_sums = self._episode_term_sums[env_ids].detach()
        if term_sums.numel() == 0:
            return
        target_sums = self._extract_target(term_sums).detach()
        self._episode_term_history.append(term_sums)
        self._episode_target_history.append(target_sums)
        self._episode_term_sums[env_ids] = 0.0

    def update(self, env):
        if not self.enabled:
            return None
        metrics: dict[str, float] = {}
        self._update_learned_reward()
        if self._last_learned_reward_loss is not None:
            metrics["RewardLearning/learned_reward_loss"] = self._last_learned_reward_loss
        metrics.update(self._weight_metrics())
        if self._update_weights(env):
            self._episode_term_history.clear()
            self._episode_target_history.clear()
        return metrics

    def _compute_raw_terms(self, env) -> torch.Tensor:
        values = []
        with torch.no_grad():
            for term_cfg in self.term_cfgs:
                term_value = term_cfg.func(env, **term_cfg.params)
                values.append(term_value.view(self.num_envs, -1).squeeze(-1))
        return torch.stack(values, dim=1)

    def _extract_target(self, raw_terms: torch.Tensor) -> torch.Tensor:
        if not self.target_indices:
            return raw_terms.sum(dim=1)
        return raw_terms[:, self.target_indices].sum(dim=1)

    def _store_obs_target(self, obs: torch.Tensor, target: torch.Tensor):
        if obs.shape[0] == 0:
            return
        samples_per_step = int(self.cfg.learned_reward_samples_per_step)
        if samples_per_step > 0 and obs.shape[0] > samples_per_step:
            indices = torch.randperm(obs.shape[0], device=obs.device)[:samples_per_step]
            obs = obs[indices]
            target = target[indices]

        num_samples = obs.shape[0]
        capacity = self._obs_buffer.shape[0]
        if num_samples > capacity:
            indices = torch.randperm(num_samples, device=obs.device)[:capacity]
            obs = obs[indices]
            target = target[indices]
            num_samples = obs.shape[0]

        end = self._buffer_pos + num_samples
        if end <= capacity:
            self._obs_buffer[self._buffer_pos:end] = obs
            self._target_buffer[self._buffer_pos:end] = target
        else:
            first = capacity - self._buffer_pos
            self._obs_buffer[self._buffer_pos:] = obs[:first]
            self._target_buffer[self._buffer_pos:] = target[:first]
            remaining = num_samples - first
            self._obs_buffer[:remaining] = obs[first:]
            self._target_buffer[:remaining] = target[first:]
        self._buffer_pos = (self._buffer_pos + num_samples) % capacity
        if num_samples > 0 and end >= capacity:
            self._buffer_full = True

    def _sample_buffer(self):
        max_size = self._obs_buffer.shape[0] if self._buffer_full else self._buffer_pos
        if max_size < self.cfg.learned_reward_batch_size:
            return None, None
        indices = torch.randint(0, max_size, (self.cfg.learned_reward_batch_size,), device=self.device)
        return self._obs_buffer[indices], self._target_buffer[indices]

    def _update_learned_reward(self):
        if self.model is None:
            return
        loss_total = 0.0
        loss_steps = 0
        for _ in range(int(self.cfg.learned_reward_epochs)):
            obs, target = self._sample_buffer()
            if obs is None:
                self._last_learned_reward_loss = None
                return
            pred = self.model(obs).squeeze(-1)
            loss = F.mse_loss(pred, target.squeeze(-1))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.learned_reward_max_grad_norm)
            self.optimizer.step()
            loss_total += float(loss.item())
            loss_steps += 1
        if loss_steps > 0:
            self._last_learned_reward_loss = loss_total / loss_steps
        else:
            self._last_learned_reward_loss = None

    def _update_weights(self, env) -> bool:
        if not self._episode_term_history:
            return False
        term_sums = torch.cat(self._episode_term_history, dim=0)
        target_sums = torch.cat(self._episode_target_history, dim=0)
        if term_sums.shape[0] < self.cfg.min_episodes:
            return False

        term_mean = term_sums.mean(dim=0)
        term_std = term_sums.std(dim=0).clamp_min(1e-6)
        target_mean = target_sums.mean()
        target_std = target_sums.std().clamp_min(1e-6)

        term_norm = (term_sums - term_mean) / term_std
        target_norm = (target_sums - target_mean) / target_std
        corr = (term_norm * target_norm.unsqueeze(1)).mean(dim=0)

        self.weights = self.weights + self.cfg.weight_lr * corr
        self.weights = self.weights - self.cfg.weight_lr * self.cfg.weight_l2 * (self.weights - self.initial_weights)
        self.weights = self.weights.clamp(-self.cfg.weight_clip, self.cfg.weight_clip)

        for name, weight in zip(self.term_names, self.weights):
            term_cfg = env.reward_manager.get_term_cfg(name)
            term_cfg.weight = float(weight.item())
        return True

    def _weight_metrics(self) -> dict[str, float]:
        metrics = {
            "RewardLearning/weight_mean": float(self.weights.mean().item()),
            "RewardLearning/weight_min": float(self.weights.min().item()),
            "RewardLearning/weight_max": float(self.weights.max().item()),
            "RewardLearning/weight_l2": float(torch.norm(self.weights).item()),
        }
        if getattr(self.cfg, "log_per_term", False):
            for name, weight in zip(self.term_names, self.weights):
                metrics[f"RewardLearning/weight/{name}"] = float(weight.item())
        return metrics
