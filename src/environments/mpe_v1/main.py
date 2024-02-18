from pettingzoo.mpe import simple_adversary_v3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ...utils.epsilon_decay import ExponentialDecay
from ...utils.utils import save_results
from ...dqn.agents.idqn import IDQNAgent
from .simulate import MPEEnvironment

seed = 42
states = 10
actions = 5
hidden = 32
sync = 5
alpha = 0.001
buffer_size = 10000
batch_size = 64

agent_dqns = ['agent_0', 'agent_1']
idqn = IDQNAgent(states, actions, hidden, seed, sync, alpha, buffer_size, batch_size)
agents = {}
# for agnt_id in agent_dqns:
#     agents[agnt_id] = idqn

generator = lambda : simple_adversary_v3.parallel_env(render_mode=None, N=2, max_cycles=50)
gamma = 0.95
tau = 0.95
decay = ExponentialDecay(0.005)

sim = MPEEnvironment(generator, agents, gamma, tau, decay, seed)
episodes = 500
rh, eh = sim.train(episodes)
print(eh)

# path  = f'{Path(__file__).resolve().parent.parent}/res/figures'
agents = {agnt_id : f'idqn_{i}' for i, agnt_id in enumerate(sim.agents.keys())}
width = 1

mean_rewards = {}
std_rewards = {}
for k, v in rh.items():
    mean_rewards[k] = np.array(v).mean(axis=1)
    std_rewards[k] = np.array(v).std(axis=1)

for k, v in mean_rewards.items():
    plt.plot(v, label=k)
plt.legend()


plt.savefig(f'xyz.png')
