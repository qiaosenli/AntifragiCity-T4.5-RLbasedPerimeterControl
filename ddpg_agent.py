"""DDPG agent, replay memory, reference/legacy updates and checkpoint IO."""
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn


def mlp(inputs, outputs):
    network = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.05)
            nn.init.zeros_(layer.bias)
    return network


class DDPG:
    """CPU agent with Gaussian exploration and soft target updates per batch."""

    def __init__(self, state_dim, action_low=(0.1,), action_high=(0.9,), *,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 capacity=10000, batch_size=64, seed=0, normalize_critic_actions=False):
        low, high = np.asarray(action_low, dtype=np.float32), np.asarray(action_high, dtype=np.float32)

        self.config = dict(state_dim=state_dim, action_low=low.tolist(), action_high=high.tolist(),
                           actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
                           capacity=capacity, batch_size=batch_size, seed=seed,
                           normalize_critic_actions=normalize_critic_actions)
        self.normalize_critic_actions = normalize_critic_actions
        self.state_dim, self.action_dim = state_dim, low.size
        self.low, self.high = torch.tensor(low), torch.tensor(high)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.rng = np.random.default_rng(seed)
        # Do not change the caller's global Torch random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = mlp(state_dim, self.action_dim)
            self.critic = mlp(state_dim + self.action_dim, 1)
        self.target_actor, self.target_critic = deepcopy(self.actor), deepcopy(self.critic)
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.memory = deque(maxlen=capacity)
        self.reference_rounds = 0

    def _vector(self, value, size):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f'Expected a finite vector of shape ({size},)')
        return vector.copy()

    def _policy(self, network, states):
        return self.low + (torch.tanh(network(states)) + 1) * (self.high - self.low) / 2

    def _critic_input(self, states, actions):
        # Actor, replay and environment retain seconds; only critic inputs are scaled.
        if self.normalize_critic_actions:
            actions = 2 * (actions - self.low) / (self.high - self.low) - 1
        return torch.cat((states, actions), dim=1)

    def choose_action(self, state, noise_std=0.0):
        """Return an ordered action vector; zero noise gives evaluation behavior."""
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative')
        state = torch.tensor(self._vector(state, self.state_dim))
        with torch.no_grad():
            action = self._policy(self.actor, state).numpy()
        if noise_std:
            action = action + self.rng.normal(0, noise_std, self.action_dim)
        return np.clip(action, self.low.numpy(), self.high.numpy()).astype(np.float32)

    def remember(self, state, action, reward, next_state, terminated=False):
        """Store a transition; terminal means an MDP terminal, not a time limit."""
        state = self._vector(state, self.state_dim)
        action = self._vector(action, self.action_dim)
        next_state = self._vector(next_state, self.state_dim)
        if np.any(action < self.low.numpy()) or np.any(action > self.high.numpy()):
            raise ValueError('Replay action lies outside configured bounds')
        if not np.isfinite(reward) or not isinstance(terminated, (bool, np.bool_)):
            raise ValueError('Reward must be finite and terminated must be boolean')
        self.memory.append((state, action, float(reward), next_state, bool(terminated)))

    @torch.no_grad()
    def _targets(self, rewards, next_states, terminals):
        next_actions = self._policy(self.target_actor, next_states)
        next_q = self.target_critic(self._critic_input(next_states, next_actions))
        return rewards + self.gamma * (1 - terminals) * next_q

    def train_step(self):
        """One replay batch; return losses, or None until enough samples exist."""
        if len(self.memory) < self.batch_size:
            return None
        indices = self.rng.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[int(i)] for i in indices]
        states, actions, rewards, next_states, terminals = (
            torch.tensor(np.asarray(column), dtype=torch.float32) for column in zip(*batch))
        targets = self._targets(rewards[:, None], next_states, terminals[:, None])
        predicted = self.critic(self._critic_input(states, actions))
        critic_loss = nn.functional.mse_loss(predicted, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        try:
            actor_loss = -self.critic(self._critic_input(states, self._policy(self.actor, states))).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for target, online in [(self.target_actor, self.actor), (self.target_critic, self.critic)]:
                for target_weight, weight in zip(target.parameters(), online.parameters()):
                    target_weight.lerp_(weight, self.tau)
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def train_reference_round(self, *, train_sample_size=1000, critic_epochs=128,
                              critic_batch_size=256, critic_validation_fraction=0.2,
                              critic_patience=20, actor_steps=2, target_update_period=5):
        """Fit a critic on one fixed replay sample, then update the actor separately.

        Targets and train/validation membership stay fixed for this round. Early
        stopping retains the last fitted critic (no best-weight restoration).
        Fewer than two samples cannot form a holdout, so no training is performed.
        """
        if min(train_sample_size, critic_epochs, critic_batch_size, critic_patience,
               actor_steps, target_update_period) < 1 or not 0 < critic_validation_fraction < 1:
            raise ValueError('Use positive training counts and validation fraction in (0, 1)')
        n_sample = min(train_sample_size, len(self.memory))
        if n_sample < 2:
            return None
        indices = self.rng.choice(len(self.memory), n_sample, replace=False)
        batch = [self.memory[int(i)] for i in indices]
        states, actions, rewards, next_states, terminals = (
            torch.tensor(np.asarray(column), dtype=torch.float32) for column in zip(*batch))
        targets = self._targets(rewards[:, None], next_states, terminals[:, None])
        critic_inputs = self._critic_input(states, actions)
        validation_size = min(n_sample - 1, max(1, int(n_sample * critic_validation_fraction)))
        train_size = n_sample - validation_size
        batch_size = min(critic_batch_size, train_size)
        best_validation, stale_epochs, critic_steps = float('inf'), 0, 0
        history = []
        for epoch in range(critic_epochs):
            order = self.rng.permutation(train_size)
            loss_sum = 0.0
            for start in range(0, train_size, batch_size):
                selected = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
                prediction = self.critic(critic_inputs[selected])
                loss = nn.functional.mse_loss(prediction, targets[selected])
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                critic_steps += 1
                loss_sum += loss.item() * len(selected)
            training_loss = loss_sum / train_size
            with torch.no_grad():
                validation_loss = nn.functional.mse_loss(
                    self.critic(critic_inputs[train_size:]), targets[train_size:]).item()
            history.append(dict(epoch=epoch + 1, training_loss=training_loss,
                                validation_loss=validation_loss))
            if validation_loss < best_validation:
                best_validation, stale_epochs = validation_loss, 0
            else:
                stale_epochs += 1
                if stale_epochs >= critic_patience:
                    break

        actor_losses = []
        self.critic.requires_grad_(False)
        try:
            for _ in range(actor_steps):
                current_actions = self._policy(self.actor, states)
                loss = -self.critic(self._critic_input(states, current_actions)).mean()
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(loss.item())
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            predicted = self.critic(critic_inputs)
        self.reference_rounds += 1
        hard_update = self.reference_rounds % target_update_period == 0
        if hard_update:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        return dict(
            replay_buffer_size=len(self.memory), fixed_sample_size=n_sample,
            critic_train_size=train_size, critic_validation_size=validation_size,
            critic_epochs_completed=len(history), critic_optimizer_steps=critic_steps,
            actor_optimizer_steps=len(actor_losses), critic_training_loss=training_loss,
            critic_validation_loss=validation_loss, best_critic_validation_loss=best_validation,
            actor_losses=actor_losses, actor_loss_final=actor_losses[-1],
            actor_loss_mean=float(np.mean(actor_losses)),
            actor_loss=float(np.mean(actor_losses)), critic_loss=training_loss,
            mean_predicted_q=predicted.mean().item(), mean_target_value=targets.mean().item(),
            max_abs_predicted_q=predicted.abs().max().item(),
            max_abs_target_value=targets.abs().max().item(),
            hard_target_update=hard_update, reference_rounds_completed=self.reference_rounds,
            critic_epoch_history=history)

    def save(self, path):
        """Save models/optimizers and exploration RNG; replay memory is excluded."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: getattr(self, name).state_dict() for name in
                ('actor', 'critic', 'target_actor', 'target_critic', 'actor_optimizer', 'critic_optimizer')}
        torch.save(dict(config=self.config, rng=self.rng.bit_generator.state,
                        reference_rounds=self.reference_rounds, **data), path)

    @classmethod
    def load(cls, path):
        """Restore checkpoint with an empty replay buffer; refill before training."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        agent = cls(**data.pop('config'))
        agent.rng.bit_generator.state = data.pop('rng')
        agent.reference_rounds = data.pop('reference_rounds', 0)
        for name, state in data.items():
            getattr(agent, name).load_state_dict(state)
        return agent
