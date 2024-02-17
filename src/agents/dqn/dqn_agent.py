import random
import numpy as np
import torch
import torch.optim as optim

from ...network.q_network import QNetwork
from ...buffer.replay_buffer import ReplayBuffer
from ..agent import Agent

class DQNAgent(Agent):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size = 32,
                 seed=42,
                 sync=5,
                 alpha = 0.001,
                 batch_size=32
                 ):
        """
        initialize a DQNAgent with the given settings.

        Parameters:
        - sync (int): timesteps between target network updates
        - alpha (float): learning rate for the optimizer
        """
        
        super().__init__(state_size, action_size, seed)

        # networks
        self.qnetwork_policy = QNetwork(state_size, action_size, hidden_size=hidden_size, seed=seed)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size=hidden_size, seed=seed)
        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=alpha) # hardcoded adam

        # replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size=1000, batch_size=batch_size, seed=seed)
        self.timestep = 0
        self.sync = sync

    def step(self, state, action, reward, next_state, done, gamma, tau):
        """
        process a step taken in the environment, save the experience in the replay buffer,
        and periodically update the model by learning from a batch of experiences.
        this method increments a timestep counter every call, and every `self.sync` steps,
        it checks if the replay buffer has enough experiences to sample a batch. If so,
        it samples a batch of experiences and calls the `learn` method to update the model.

        Parameters:
        - state (array-like): the current state of the environment.
        - action (int): the action taken in the current state.
        - reward (float): the reward received after taking the action.
        - next_state (array-like): the state of the environment after taking the action.
        - done (bool): a flag indicating whether the episode has ended (True if the episode is done, False otherwise).
        """
        # add memory into replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # sync
        self.timestep = (self.timestep + 1) % self.sync
        if self.timestep == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma, tau)

    def act(self, state, action_mask, **kwargs):
        """
        Select an action for the given state using an epsilon-greedy policy,
        considering only valid actions as specified by an action mask when exploring.

        Parameters:
        - state (np.ndarray): the current state of the environment.
        - action_mask (np.ndarray): a binary mask of valid actions, where 1 indicates a valid action.
        - eps (float): The epsilon value for epsilon-greedy action selection.

        Returns:
        - int: the selected action index.
        """
        
        if 'eps' in kwargs: eps = kwargs['eps'] 
        else: raise KeyError('eps not in kwargs')

        with torch.no_grad():
            action_values = self.qnetwork_policy(state)

        # eps-greedy action selection
        if random.random() > eps:
            return action_values.argmax().item()
        else:
            possible_actions = np.arange(len(action_mask))[action_mask == 1]
            return random.choice(possible_actions)

    def learn(self, experiences, gamma, tau):
        """
        train the policy network using a batch of experience tuples and 
        softly update the target network parameters.

        this method executes several key steps:
        1. it calculates the maximum predicted q values for the next states (q_targets_next)
        from the target network, without gradient tracking.
        2. it computes the Q targets for the current states using the Bellman equation,
        which are then used as the ground truth during the loss calculation.
        3. it determines the expected Q values from the policy network based on the
        actual actions taken.
        4. it calculates the loss as the mean squared error between the expected Q values
        and the computed Q targets.
        5. it performs backpropagation to update the policy network.
        6. it softly updates the target network parameters using the tau parameter.

        Parameters:
        - experiences (Tuple[torch.Tensor]): batch of experience tuples (s, a, r, s', done).
        - gamma (float): discount factor for future rewards.
        - tau (float): interpolation parameter for soft updating the target network parameters.
        """
        states, actions, rewards, next_states, dones = experiences

        # max_a' q(s', a')
        with torch.no_grad():
            q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
        # q from targets and policy for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_policy(states).gather(1, actions)

        loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target model
        self.soft_update(self.qnetwork_policy, self.qnetwork_target, tau)

    def soft_update(self, policy_model, target_model, tau):
        """
        soft update model parameters.
        target_model = target_model + tau * (policy_model - target_model)

        Parameters:
        policy_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
        """
        with torch.no_grad():
            for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
                target_param.copy_(target_param.data + tau*(policy_param.data - target_param.data))
        