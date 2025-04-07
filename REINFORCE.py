import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Beta, Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, std_init=0.5):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)

        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(std_init))

        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu = torch.tanh(self.mu_head(x))
        return mu, self.log_std

    def getDist(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist


class Agent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim,
        gamma,
        lr,
        entropy_coef,
        device=None,
        lr_decay=0.99,
        lr_decay_steps=1000,
        baseline_buffer_size=100000,
    ):

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        self.actor = Actor(observation_dim, action_dim, hidden_dim).to(self.device)

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        self.baseline = None
        self.baseline_buffer_size = baseline_buffer_size

        self.trajectory_returns = deque(maxlen=baseline_buffer_size)

        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=0.001
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_steps, gamma=lr_decay
        )

        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.episode_count = 0

    def act(self, observation, deterministic=False):
        """Select action based on the current policy"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        if deterministic:
            with torch.no_grad():
                action_mean, _ = self.actor(state)
                return action_mean.cpu().numpy().flatten()
        else:

            dist = self.actor.getDist(state)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)

            return action.cpu().detach().numpy().flatten()

    def collect(self, reward, log_prob):
        """Store reward from the environment"""
        self.rewards.append(reward)

    def learn(self):
        self.episode_count += 1
        """Update policy using REINFORCE algorithm with entropy regularization"""

        if len(self.rewards) == 0:
            return

        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        self.trajectory_returns.append(returns.mean().item())

        if (
            len(self.trajectory_returns) > 0
            and self.episode_count % 10 == 0
            and self.baseline is not None
        ):

            discount_factor = 0.9999
            weights = np.array(
                [
                    discount_factor ** (len(self.trajectory_returns) - i - 1)
                    for i in range(len(self.trajectory_returns))
                ]
            )
            weights = weights / weights.sum()

            weighted_returns = [r * w for r, w in zip(self.trajectory_returns, weights)]
            baseline = sum(weighted_returns)
            if baseline > self.baseline:

                self.baseline = torch.tensor(
                    baseline * 0.05 + self.baseline * 0.95, device=self.device
                )
            else:

                self.baseline = torch.tensor(
                    baseline * 0.05 + self.baseline * 0.95, device=self.device
                )
        else:
            if self.baseline is None:
                self.baseline = returns.mean()

        advantages = returns - self.baseline

        policy_loss = 0
        entropy_sum = 0

        for i, (state, action, log_prob, advantage) in enumerate(
            zip(self.states, self.actions, self.log_probs, advantages)
        ):

            dist = self.actor.getDist(state)

            entropy = dist.entropy().mean()
            entropy_sum += entropy

            policy_loss += -log_prob * advantage

        entropy_mean = (
            entropy_sum / len(self.states)
            if self.states
            else torch.tensor(0.0).to(self.device)
        )
        policy_loss = (
            policy_loss / len(self.states)
            if self.states
            else torch.tensor(0.0).to(self.device)
        )

        loss = policy_loss - self.entropy_coef * entropy_mean

        print(
            "Returns Mean: ",
            returns.mean().item(),
            "Baseline: ",
            self.baseline.item(),
            "Policy Loss: ",
            policy_loss.item(),
            "Entropy: ",
            entropy_mean.item(),
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        return policy_loss.item()
