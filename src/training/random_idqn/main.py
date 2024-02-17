import numpy as np
import torch
import random
from typing import Dict, Optional

from ...dqn.agents.idqn import IDQNAgent
from ...agents.random import RandomAgent
from typing import List, Tuple, Dict
from ...utils.epsilon_decay import ExponentialDecay

class IDQNRandomEnvironment:
    """
    An environment class that integrates an IDQN agent with random agents
    for training in a mixed agent setting.

    Attributes:
        env: Function to instantiate PettingZoo Environment.
        idqn (IDQNAgent): The IDQN agent to be trained.
        idqn_id (Optional[str]): Identifier for the IDQN agent, can be None.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update parameter for target network updates.
        decay: Decay strategy for epsilon, affecting exploration.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, generator, idqn: IDQNAgent, idqn_id: Optional[str] = None,
                 gamma: float = 0.99, tau: float = 0.99,
                 decay=ExponentialDecay(0.001), seed: int = 42):
        
        self.generator = generator
        self.env = generator()
        self.env.reset()

        self.gamma = gamma
        self.tau = tau
        self.decay = decay
        self.seed = seed
        random.seed(self.seed)

        self.idqn = idqn
        idqn_id = idqn_id if idqn_id is not None else random.choice(self.env.agents)
        self.idqn.id(idqn_id)
        self.agents = {idqn_id: idqn}
        for agnt_id in (a for a in self.env.agents if a != idqn_id):
            agent = RandomAgent(seed)
            agent.id(agnt_id)
            self.agents[agnt_id] = agent

    def train(self, episodes: int) -> Dict[str, np.array]:
        """
        Trains the IDQN and random agents for a specified number of episodes.

        Args:
            episodes (int): The number of episodes to train the agents.

        Returns:
            A dictionary with agent IDs as keys and their reward histories as values.
        """
        reward_history = {agnt_id: np.zeros(episodes) for agnt_id in self.env.agents}
        prev_state_action = (None, None)

        epsilon = 1  # Start with a fully exploratory policy
        epsilon_history = np.zeros(episodes)

        for i in range(episodes):
            
            self.env = self.generator()
            self.env.reset()

            for agnt_id in self.env.agent_iter():
                observation, reward, terminated, truncated, _ = self.env.last()
                done = terminated or truncated
                obs, action_mask = observation.values()
                state = torch.flatten(torch.tensor(obs, dtype=torch.float32))
                agent = self.agents[agnt_id]

                if done:
                    action = None
                    reward_history[agnt_id][i] = reward
                if agent._id == self.idqn._id:
                    ps, pa = prev_state_action
                    if ps is not None and pa is not None:
                        experience = (ps, pa, reward, state, done)
                        self.idqn.step(experience, self.gamma, self.tau)
                    if not done:
                        action = self.idqn.act(state, action_mask, eps=epsilon)
                        prev_state_action = (state, action)
                else:
                    if not done:
                        action = agent.act(state, action_mask)

                self.env.step(action)

            epsilon_history[i] = epsilon
            epsilon = self.decay.get(epsilon, i)

        return reward_history, epsilon_history
