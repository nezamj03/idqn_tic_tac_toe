import numpy as np
import torch
import random
from typing import List, Tuple, Dict

from ...dqn.agents.idqn import IDQNAgent
from typing import List
from ...utils.epsilon_decay import ExponentialDecay

class IDQNEnvironment:
    """
    An environment class for Incremental Deep Q-Network (IDQN) training.

    Attributes:
        generator: function that instantiates the PettingZoo Environment.
        agents (List[IDQNAgent]): The list of IDQN agents to train.
        gamma (float): Discount factor for future rewards.
        tau (float): Soft update parameter for target network updates.
        decay: Decay strategy for epsilon, affecting exploration vs exploitation.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, generator, agents: List[IDQNAgent], gamma: float = 0.99, 
                 tau: float = 0.99, decay=ExponentialDecay(0.001), seed=42):
        """
        Initializes the IDQNEnvironment with the environment, agents, and training parameters.
        """
        self.generator =generator
        self.env = generator()
        self.env.reset()

        self.gamma = gamma
        self.tau = tau
        self.decay = decay
        self.seed = seed
        random.seed(self.seed)

        self.agents = {}
        assert len(agents) == len(self.env.agents), (
            "Size of `agents` does not match size of `env.agents`."
        )
        for agent, agent_id in zip(agents, self.env.agents):
            agent.id(agent_id)
            self.agents[agent_id] = agent

    def train(self, episodes: int) -> Tuple[Dict[str, np.array], np.array]:
        """
        Trains the IDQN and random agents for a specified number of episodes.

        Args:
            episodes (int): The number of episodes to train the agents.

        Returns:
            A tuple of two elements:
            - A dictionary with agent IDs as keys and their reward histories as values.
            - A numpy array containing the history of epsilon values for each episode.
        """
        reward_history = {agnt_id: np.zeros(episodes) for agnt_id in self.env.agents}
        prev_state_action = {agnt_id: (None, None) for agnt_id in self.env.agents}

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
                agent: IDQNAgent = self.agents[agnt_id]

                ps, pa = prev_state_action[agnt_id]
                if ps is not None and pa is not None:
                    experience = (ps, pa, reward, state, done)
                    agent.step(experience, self.gamma, self.tau)
                
                if done:
                    action = None
                    reward_history[agnt_id][i] = reward
                else:
                    action = agent.act(state, action_mask, eps=epsilon)
                    prev_state_action[agnt_id] = (state, action)

                self.env.step(action)

            epsilon_history[i] = epsilon
            epsilon = self.decay.get(epsilon, i)

        return reward_history, epsilon_history
