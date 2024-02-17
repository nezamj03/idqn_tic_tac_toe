import numpy as np
from pettingzoo.classic import tictactoe_v3

from ..agents.dqn_agent import DQNAgent
from ..agents.random_agent import RandomAgent
from ..utils.utils import *

class TicTacToeDQNRandom:

    def __init__(self, agent_dqn: DQNAgent,
                 agent_random: RandomAgent, 
                 player_dqn: str, 
                 gamma: float = 0.9, 
                 tau: float = 0.75, 
                 decay: float = 0.01):
        """
        Initializes the TicTacToeDQNRandom class with both agents and training parameters.

        Parameters:
        - agent_dqn: The DQN agent to be trained.
        - agent_random: The random agent to compete against.
        - player_dqn: Player symbol for the DQN agent ('x' or 'o').
        - gamma: Discount factor for future rewards.
        - tau: Soft update parameter for the target network.
        - decay: Rate of decay for epsilon in the epsilon-greedy policy.
        """
        self.gamma = gamma
        self.tau = tau
        self.decay = decay

        self.env = tictactoe_v3.env(render_mode=None)
        self.env.reset()

        self.agent_dqn = agent_dqn
        self.agent_random = agent_random
        self.agents = tuple(self.env.agents)

        if player_dqn == 'x':
            self.agent_dqn.id(self.env.agents[0])
            self.agent_random.id(self.env.agents[1])
        else:
            self.agent_dqn.id(self.env.agents[1])
            self.agent_random.id(self.env.agents[0])

    def train(self, episodes: int) -> tuple[dict, np.array]:
        """
        Trains the DQN agent by playing against the random agent for a specified number of episodes.

        Parameters:
        - episodes: The number of episodes to train over.

        Returns:
        - A tuple containing the reward history for each agent and the history of epsilon values used during training.
        """
        reward_history = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}
        prev_sa = (None, None)

        epsilon = 1  # Start with a fully exploratory policy
        epsilon_history = np.zeros(episodes)

        for i in range(episodes):
            self.env = tictactoe_v3.env(render_mode=None)
            self.env.reset()

            for agent_id in self.env.agent_iter():
                observation, reward, terminated, truncated, _ = self.env.last()
                done = terminated or truncated
                state, action_mask = observation.values()
                input_tensor = state_to_input(state)

                # DQN Agent's turn
                if self.agent_dqn._id == agent_id: 
                    ps, pa = prev_sa
                    if ps is not None and pa is not None:
                        self.agent_dqn.step(ps, pa, reward, input_tensor, done, self.gamma, self.tau)
                    if done: 
                        action = None
                        reward_history[agent_id][i] = reward
                    else:
                        action = self.agent_dqn.act(input_tensor, action_mask, eps=epsilon)
                    prev_sa = (input_tensor, action)
                # Random Agent's turn
                else:  
                    if done:
                        action = None
                        reward_history[agent_id][i] = reward
                    else:
                        action = self.agent_random.act(input_tensor, action_mask)

                self.env.step(action)

            epsilon *= np.exp(-self.decay * i)  # Exponential decay of epsilon
            epsilon_history[i] = epsilon

        return reward_history, epsilon_history

# Additional functions like state_to_board can be documented similarly.

if __name__ == '__main__':
    gamma = 1
    tau = 1
    decay = 0.0000015
    states = 18
    actions = 9
    seed = 42
    sync = 5

    dqn = DQNAgent(states, actions, seed=seed, sync=sync)
    rand = RandomAgent(states, actions, seed=seed)
    player = 'x'
    ttt = TicTacToeDQNRandom(dqn, rand, player, gamma, tau, decay)
    
    agents = {
        ttt.agents[0] : 'dqn' if player == 'x' else 'random',
        ttt.agents[1] : 'dqn' if player == 'o' else 'random'
    }
    
    rh, eh = ttt.train(2000)
    width = 100
    save_results(agents, rh, eh, 'initial', width=width)
    save_policy(dqn, 'initial')
