import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2
    ):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return mean, std

    def sample(self, state, deterministic=False):
        mean, std = self.forward(state)

        if deterministic:
            return torch.tanh(mean), None

        normal = Normal(mean, std)

        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - (2 * (np.log(2) - x_t - F.softplus(-2 * x_t)))
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)

        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        target_update_interval=1,
        batch_size=256,
        replay_buffer_size=1000000,
        device=None,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        self.policy = PolicyNetwork(observation_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q_network = QNetwork(observation_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.target_q_network = QNetwork(observation_dim, action_dim, hidden_dim).to(
            self.device
        )

        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(param.data)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.auto_entropy_tuning = auto_entropy_tuning

        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(
                torch.Tensor([action_dim]).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.updates = 0

    def act(self, observation, deterministic=False):
        """Select action based on the current policy"""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.policy.sample(state, deterministic=deterministic)

        return action.cpu().numpy().flatten()

    def collect(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_target_networks(self):
        """Soft update of target network"""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def learn(self):
        """Update policy and value using SAC algorithm"""

        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_q_network(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q1, current_q2 = self.q_network(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_probs = self.policy.sample(states)
        q1, q2 = self.q_network(states, new_actions)
        q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_probs - q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self.update_target_networks()

        return q_loss.item(), policy_loss.item()

    def save(self, path):
        """Save model to disk"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                "log_alpha": self.log_alpha if self.auto_entropy_tuning else None,
                "alpha_optimizer_state_dict": (
                    self.alpha_optimizer.state_dict()
                    if self.auto_entropy_tuning
                    else None
                ),
            },
            path,
        )

    def load(self, path):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

        if self.auto_entropy_tuning:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
            self.alpha = self.log_alpha.exp().item()
