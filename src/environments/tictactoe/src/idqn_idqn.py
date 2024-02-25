import numpy as np
import torch
import random
from typing import List, Tuple, Dict
from pettingzoo.classic import tictactoe_v3
from pathlib import Path

from ....dqn.agents.idqn import IDQNAgent
from ....utils.epsilon_decay import ExponentialDecay, LinearDecay
from ....utils.utils import save_results
from ....dqn.agents.idqn import IDQNAgent

class TicTacToeIDQNEnvironment:
    """
    An environment class for Independent Deep Q-Network (IDQN) training.

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
                obs, action_mask = observation.values()
                state = torch.flatten(torch.tensor(obs).float())
                agent: IDQNAgent = self.agents[agnt_id]

                ps, pa = prev_state_action[agnt_id]
                if ps is not None and pa is not None:
                    experience = (ps, pa, reward, state, (terminated or truncated))
                    agent.step(experience, self.gamma, self.tau)
                    
                if terminated or truncated:
                    action = None
                    reward_history[agnt_id][i] = reward
                else:
                    action = agent.act(state, action_mask, eps=epsilon)
                    prev_state_action[agnt_id] = (state, action)

                self.env.step(action)

            epsilon_history[i] = epsilon
            epsilon = self.decay.get(epsilon, i)

        return reward_history, epsilon_history


def main():
    seed = 42
    states = 3 * 3 * 2
    actions = 9
    hidden = 32
    sync = 5
    alpha = 0.001
    buffer_size = 500
    batch_size = 50

    agents = 2
    idqn_agents = [IDQNAgent(states, actions, hidden, seed, sync, alpha, buffer_size, batch_size)
                for _ in range(agents)]

    generator = lambda : tictactoe_v3.env(render_mode=None)
    gamma = 1
    tau = 0.95
    episodes = 5000
    decay = LinearDecay(episodes)

    sim = TicTacToeIDQNEnvironment(generator, idqn_agents, gamma, tau, decay, seed)
    rh, eh = sim.train(episodes)

    path  = f'{Path(__file__).resolve().parent.parent}/res/figures'
    agent_names = {agnt_id : f'idqn_{i}' for i, agnt_id in enumerate(sim.agents.keys())}
    width = 100

    save_results(agent_names, rh, eh, f'{path}/v0_idqn_idqn', width)

if __name__ == "__main__":
    main()