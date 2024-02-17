from pettingzoo.classic import connect_four_v3
from pathlib import Path

from ....utils.epsilon_decay import ExponentialDecay
from ....utils.utils import save_policy, save_results
from ....dqn.agents.idqn import IDQNAgent
from ....training.random_idqn.main import IDQNRandomEnvironment

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
idqn_id = None # randomly select player 1 or player 2
gamma = 0.99
tau = 0.95
decay = ExponentialDecay(0.001)

sim = IDQNRandomEnvironment(generator, agent, idqn_id, gamma, tau, decay, seed)
episodes = 5000
rh, eh = sim.train(episodes)

path  = f'{Path(__file__).resolve().parent.parent}/res/figures'
agents = {}
for agnt_id in sim.agents.keys():
    if agnt_id == agent._id: agents[agnt_id] = 'idqn'
    else: agents[agnt_id] = 'random'
width = 100

save_results(agents, rh, eh, f'{path}/v0_random_idqn', width)