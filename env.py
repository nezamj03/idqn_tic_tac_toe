from pettingzoo.classic import tictactoe_v3
from network import DQN
from torch import nn
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pyplot as plt
import random 
from utils import state_to_dqn_input, sync, plot_dictionary
from agents import DQNAgent, Random
from agents import Agent
import time

random.seed('nezam')

class TicTacToeDQL():

    def __init__(self, **kwargs):
        """ hyperparameter selection """
        self.memory_size = kwargs['memory_size'] if 'memory_size' in kwargs else float('inf')
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 25
        self.lr_alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.0001 
        self.dr_gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.9
        self.sync = kwargs['sync'] if 'sync' in kwargs else 5
        self.loss_fn = kwargs['loss'] if 'loss' in kwargs else nn.MSELoss()
        self.render = kwargs['render_mode'] if 'render_mode' in kwargs else None
        self.name = kwargs['name'] if 'name' in kwargs else 'ttt'
        self.optimizer = None
        self.env = tictactoe_v3.env(render_mode=self.render)
        self.env.reset()

    def train(self, episodes):
        """ train dqn in tic tac toe """

        num_states = 18 # this can be made more flexible in the future
        hidden_dimension = 12 # this can be made more flexible in the future
        num_actions = 9 # this can be made more flexible in the future
        policy_dqn = DQN(num_states, hidden_dimension, num_actions) # policy network 
        target_dqn = DQN(num_states, hidden_dimension, num_actions) # target network
        dqn_agent = DQNAgent(policy_network=policy_dqn, target_network=target_dqn)

        # hardcoded for now (agent, to be optimized)
        agent_map = { 
            self.env.agents[0] : (dqn_agent, True), 
            self.env.agents[1] : (Random(), False)
        }

        reward_history = { agent_id : np.zeros(episodes) for agent_id in self.env.agents }
        sa_history = { agent_id : None for agent_id in self.env.agents} 

        epsilon = 1 # start with a random action
        epsilon_history = np.zeros(episodes)

        self.optimizer = Adam(policy_dqn.parameters(), lr=self.lr_alpha)

        steps = 0 # used for syncing
        for i in range(episodes):

            self.env = tictactoe_v3.env(render_mode=self.render)
            self.env.reset()
            # make target params same as policy when a certain number of trials have been completed
            steps = sync(steps, self.sync, networks = [policy_dqn, target_dqn]) 

            for agent_id in self.env.agent_iter():

                agent = agent_map[agent_id][0]
                observation, reward, terminated, truncated, _ = self.env.last() # receives some state
                  
                terminal = terminated or truncated
                state, action_mask = observation.values()
                if terminal:
                    action = None 
                    reward_history[agent_id][i] = reward
                else:
                    action = agent.action(action_mask= action_mask, epsilon= epsilon, input= state)

                self.env.step(action)
                
                if sa_history[agent_id]:
                    agent.memory.append((*sa_history[agent_id], state, reward, terminal))
                sa_history[agent_id] = (state, action)

                steps+=1

            for agent_id, (agent, optimize) in agent_map.items():
                # when memory deque is sufficiently large, train policy network
                if optimize and len(agent.memory) > self.batch_size: 
                    batch = agent.memory.sample(self.batch_size) # THIS IS WHERE BATCH SIZE IS USED
                    self.optimize(agent, batch)        

            # linear epsilon decay
            epsilon = max(epsilon - 1/episodes, 0)
            epsilon_history[i] = epsilon

        self.env.close()

        torch.save(policy_dqn.state_dict(), f"{self.name}.pt")
        plt.figure(1)

        # average rewards vs episodes
        cum_rewards = {agent_id : np.zeros(episodes) for agent_id in agent_map.keys()}
        for agent_id in cum_rewards.keys():
            for i in range(episodes):
                cum_rewards[agent_id][i] = np.mean(reward_history[agent_id][max(0, i-100):(i+1)])
        plt.subplot(121)
        plot_dictionary(cum_rewards)
        
        # epsilon decay vs episodes
        plt.subplot(122)
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig(f'{self.name}.png')
        plt.close()

    def optimize(self, agent : DQNAgent, batch):

        current_q_list = []
        target_q_list = []

        for state, action, next_state, reward, terminal in batch:

            if terminal: 
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.dr_gamma * agent.target_network(state_to_dqn_input(next_state)).max()
                    )

            # Get the current set of Q values
            current_q_list.append(agent.policy_network(state_to_dqn_input(state)))
            # Get the target set of Q values
            target_q = agent.target_network(state_to_dqn_input(state)) 
            # replace target action-value with immediate reward + discounted fut
            target_q[action] = target
            target_q_list.append(target_q)
                
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # backprop given the loss from the samples
        # note the loss is mse, so is averaged
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    # batchsizes = [2, 5, 10, 20, 50]
    # memories = [20, 50, 100, 500, float('inf')]
    # durations = {}
    # for bs in batchsizes:
    #     for memory in memories:
    #         if bs < memory:
    #             t1 = time.time()
    #             name = f'batchsize: {bs}, memory_size : {memory}'
    #             print(name)
    #             ttt = TicTacToeDQL(batch_size = bs, memory_size = memory, name = name)
    #             ttt.train(5000)
    #             t2 = time.time()
    #             durations[(bs, memory)] = np.round(t2 - t1, 3)
    # print(durations)
    ttt = TicTacToeDQL(batch_size = 15, sync = 4, name = 'batchsize100.sync4.memoryNone')
    ttt.train(10000)