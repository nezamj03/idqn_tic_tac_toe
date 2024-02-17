import random
import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Tuple

from ..network.q_network import QNetwork
from ..replaybuffer.replay_buffer import ReplayBuffer
from ...agents.agent import Agent

class IDQNAgent(Agent):
    """
    Implements an Incremental Deep Q-Network (IDQN) Agent.

    Attributes:
        state_size (int): Dimension of each state.
        action_size (int): Dimension of each action.
        hidden_size (int): Number of nodes in the hidden layers.
        seed (int): Random seed for reproducibility.
        sync (int): How often to sync the target network.
        alpha (float): Learning rate.
        buffer_size (int): Size of the replay buffer.
        batch_size (int): Sample size from replay buffer.
        replay_buffer (Optional[ReplayBuffer]): The replay buffer for experience replay.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 32, 
                 seed: int = 42, sync: int = 5, alpha: float = 0.001, 
                 buffer_size: int = 1000, batch_size: int = 32, 
                 replay_buffer: Optional[ReplayBuffer] = None):
        super().__init__(seed)
        self.memory = replay_buffer if replay_buffer is not None else ReplayBuffer(
            action_size, buffer_size=buffer_size, batch_size=batch_size, seed=seed)

        # Networks
        self.policy = QNetwork(state_size, action_size, hidden_size=hidden_size, seed=seed)
        self.target = QNetwork(state_size, action_size, hidden_size=hidden_size, seed=seed)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)  # Adam optimizer

        # Replay buffer
        self.timestep = 0
        self.sync = sync

    def step(self, experience: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                     torch.Tensor, torch.Tensor], gamma: float, tau: float):
        """
        Takes a step in the environment to update the replay buffer and possibly learn from experiences.

        Args:
            experience (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
                A tuple of tensors representing the experience (state, action, reward, next_state, done).
            gamma (float): Discount factor for future rewards.
            tau (float): Interpolation parameter for soft update of target network.
        """
        self.memory.add(*experience)
        
        self.timestep = (self.timestep + 1) % self.sync
        if self.timestep == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, gamma, tau)

    def act(self, state: torch.Tensor, action_mask: np.ndarray, eps: float) -> int:
        """
        Returns actions for given state as per current policy.

        Args:
            state (torch.Tensor): Current state.
            action_mask (np.ndarray): A mask indicating valid actions.
            eps (float): The epsilon for epsilon-greedy action selection.

        Returns:
            int: The selected action.
        """
        if random.random() > eps:
            with torch.no_grad():
                action_values = self.policy(state)
            return action_values.argmax().item()
        else:
            possible_actions = np.flatnonzero(action_mask == 1)
            return random.choice(possible_actions)

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                       torch.Tensor, torch.Tensor], gamma: float, tau: float):
        """
        Update policy and value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor, ...]): Tuple of experiences (s, a, r, s', done).
            gamma (float): Discount factor.
            tau (float): Interpolation parameter for soft update.
        """
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            q_targets_next = self.target(next_states).max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.policy(states).gather(1, actions)

        # Compute loss
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        self.soft_update(tau)

    def soft_update(self, tau: float):
        """
        Soft update model parameters of the target network using those of the policy network.

        Args:
            tau (float): Interpolation parameter.
        """

        with torch.no_grad():
            for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
                target_param.copy_(target_param.data + tau*(policy_param.data - target_param.data))
        