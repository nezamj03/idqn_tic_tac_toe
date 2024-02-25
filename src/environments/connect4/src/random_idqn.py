import numpy as np
import torch
import random
from typing import Tuple, Dict, Optional
from pettingzoo.classic import connect_four_v3
from pathlib import Path

from ....dqn.agents.idqn import IDQNAgent
from ....agents.random import RandomAgent
from ....utils.epsilon_decay import ExponentialDecay, LinearDecay
from ....utils.utils import save_results
from ....dqn.agents.idqn import IDQNAgent

class ConnectFourIDQNRandomEnvironment:
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
        self.idqn_id = idqn_id if idqn_id is not None else random.choice(self.env.agents)
        self.agents = {self.idqn_id: idqn}
        for agnt_id in (a for a in self.env.agents if a != self.idqn_id):
            agent = RandomAgent(seed)
            self.agents[agnt_id] = agent

    def train(self, episodes: int) -> Tuple[Dict[str, np.array], np.array]:
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
                obs, action_mask = observation.values()
                state = torch.flatten(torch.tensor(obs).float())
                agent = self.agents[agnt_id]

                if terminated or truncated:
                    action = None
                    reward_history[agnt_id][i] = reward
                if agnt_id == self.idqn_id:
                    ps, pa = prev_state_action
                    if ps is not None and pa is not None:
                        experience = (ps, pa, reward, state, (terminated or truncated))
                        self.idqn.step(experience, self.gamma, self.tau)
                    if not (terminated or truncated):
                        action = self.idqn.act(state, action_mask, eps=epsilon)
                        prev_state_action = (state, action)
                elif not (terminated or truncated):
                    action = agent.act(state, action_mask)

                self.env.step(action)

            epsilon_history[i] = epsilon
            epsilon = self.decay.get(epsilon, i)

        return reward_history, epsilon_history

def main():
    seed = 42
    states = 6 * 7 * 2
    actions = 7
    hidden = 32
    sync = 5
    alpha = 0.001
    buffer_size = 10000
    batch_size = 64

    agent = IDQNAgent(states, actions, hidden, seed, sync, alpha, buffer_size, batch_size)

    generator = lambda : connect_four_v3.env(render_mode=None)
    idqn_id = None # randomly select player 1 or 2
    gamma = 1
    tau = 0.95
    episodes = 2000
    decay = LinearDecay(episodes)
    
    sim = ConnectFourIDQNRandomEnvironment(generator, agent, idqn_id, gamma, tau, decay, seed)
    rh, eh = sim.train(episodes)

    path  = f'{Path(__file__).resolve().parent.parent}/res/figures'
    agent_names = {}
    for agnt_id in sim.agents.keys():
        if agnt_id == sim.idqn_id: agent_names[agnt_id] = 'idqn'
        else: agent_names[agnt_id] = 'random'
    width = 100

    save_results(agent_names, rh, eh, f'{path}/v0_random_idqn', width)

if __name__ == "__main__":
    main()