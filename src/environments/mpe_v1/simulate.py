import numpy as np
import torch
import random

from ...agents.random import RandomAgent
from ...utils.epsilon_decay import ExponentialDecay
from ...agents.agent import Agent

class MPEEnvironment:

    def __init__(self, generator, agents,
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

        rand_agnt = RandomAgent(seed)
        self.agents = {agnt_id : agents.get(agnt_id, rand_agnt) for agnt_id in self.env.agents}
        for agnt_id, agnt in self.agents.items():
            agnt._id = agnt_id
            

    def train(self, episodes: int):
        
        reward_history = {agnt_id: [] for agnt_id in self.env.agents}

        epsilon = 1  # Start with a fully exploratory policy
        epsilon_history = np.zeros(episodes)

        for i in range(episodes):
            
            self.env = self.generator()
            observations, _ = self.env.reset()
            done = False
            prev_states = {agnt_id: None for agnt_id in self.env.agents}
            prev_actions = {agnt_id: None for agnt_id in self.env.agents}
            agent_rewards = {agnt_id: np.zeros(self.env.aec_env.max_cycles) for agnt_id in self.env.agents}
            cycle = 0

            while self.env.agents:
                
                actions = {}
                for agnt_id in self.env.agents:
                    
                    state = torch.flatten(torch.tensor(observations[agnt_id]).float())
                    agent = self.agents[agnt_id]
                
                    ps, pa = prev_states[agnt_id], prev_actions[agnt_id]
                    if ps is not None and pa is not None:
                        reward = rewards[agnt_id]
                        experience = (ps, pa, reward, state, done)
                        try: agent.step(experience, self.gamma, self.tau)
                        except: pass
                    
                    valid_actions = np.ones(self.env.action_space(agnt_id).n)

                    if done:
                        action = None
                    else:
                        action = agent.act(state, valid_actions, eps=epsilon)
                        prev_states[agnt_id] = state
                        prev_actions[agnt_id] = action

                    actions[agnt_id] = action

                observations, rewards, terminated, truncated, _ = self.env.step(actions)

                done = any(terminated[a] or truncated[a] for a in terminated.keys())
                for agnt_id in self.env.agents: agent_rewards[agnt_id][cycle] = rewards[agnt_id]
                # print(agent_rewards)
                cycle += 1

            for agnt_id in reward_history.keys(): reward_history[agnt_id].append(agent_rewards[agnt_id])
            epsilon_history[i] = epsilon
            epsilon = self.decay.get(epsilon, i)
            self.env.close() 
       
        return reward_history, epsilon_history
