import numpy as np
from pettingzoo.classic import tictactoe_v3

from ..eval.eps_decay import ExponentialDecay, LinearDecay
from ..agents.dqn_agent import DQNAgent
from ..utils.utils import *

class TicTacToeDQNDQN:

    def __init__(self, 
                 agent_dqnx: DQNAgent,
                 agent_dqno: DQNAgent, 
                 gamma: float = 0.9, 
                 tau: float = 0.75, 
                 decay = ExponentialDecay(0.001)):
       
        self.gamma = gamma
        self.tau = tau
        self.decay = decay

        self.env = tictactoe_v3.env(render_mode=None)
        self.env.reset()

        self.agent_dqnx = agent_dqnx
        self.agent_dqno = agent_dqno

        self.agent_dqnx.id(self.env.agents[0])
        self.agent_dqno.id(self.env.agents[1])

        self.agent_map = {
            self.agent_dqnx._id : self.agent_dqnx,
            self.agent_dqno._id : self.agent_dqno
        }
        
        self.agents = tuple(self.env.agents)

    def train(self, episodes: int) -> tuple[dict, np.array]:

        reward_history = {agent_id: np.zeros(episodes) for agent_id in self.env.agents}
        prev_sa = {agent_id: (None, None) for agent_id in self.env.agents}

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
                agent = self.agent_map[agent_id]

                ps, pa = prev_sa[agent_id]
                if ps is not None and pa is not None:
                    agent.step(ps, pa, reward, input_tensor, done, self.gamma, self.tau)
                if done: 
                    action = None
                    reward_history[agent_id][i] = reward
                else:
                    action = agent.act(input_tensor, action_mask, eps=epsilon)
                prev_sa[agent_id] = (input_tensor, action)
                self.env.step(action)

            epsilon_history[i] = epsilon
            epsilon = self.decay.get(epsilon, i)

        return reward_history, epsilon_history

# Additional functions like state_to_board can be documented similarly.

if __name__ == '__main__':
    
    gamma = 0.99
    tau = 0.75
    states = 18
    actions = 9
    seed = 42
    sync = 5
    hidden = 32
    batch = 96
    episodes = 7500
    decay = LinearDecay(episodes)

    dqnx = DQNAgent(states, actions, hidden_size=hidden, seed=seed, sync=sync, batch_size=batch)
    dqno = DQNAgent(states, actions, hidden_size=hidden, seed=seed, sync=sync, batch_size=batch)
    ttt = TicTacToeDQNDQN(dqnx, dqno, gamma, tau, decay)
    
    agents = {
        ttt.agents[0] : 'dqn_x',
        ttt.agents[1] : 'dqn_o'
    }
    
    rh, eh = ttt.train(episodes)
    width = 100
    for agent_id, history in rh.items():
            cp = find_convergence_point(history)
            print(f'{agents[agent_id]}_cnvg: {cp}')
            print(f'{agents[agent_id]}_cnvg_val: {history[cp:].mean() * (cp != -1) - (cp == -1)}')
            print(f'{agents[agent_id]}_cnvg_val: {history[-width:].mean()}')

    save_results(agents, rh, eh, 'initial', width=width)
