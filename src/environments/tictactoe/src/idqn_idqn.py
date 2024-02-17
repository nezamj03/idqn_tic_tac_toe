from pettingzoo.classic import tictactoe_v3
from pathlib import Path

from ....utils.epsilon_decay import ExponentialDecay
from ....utils.utils import save_policy, save_results
from ....dqn.agents.idqn import IDQNAgent
from ....training.idqn_idqn.main import IDQNEnvironment

seed = 42
states = 3 * 3 * 2
actions = 9
hidden = 32
sync = 5
alpha = 0.001
buffer_size = 10000
batch_size = 64

agents = 2
idqn_agents = [IDQNAgent(states, actions, hidden, seed, sync, alpha, buffer_size, batch_size)
               for _ in range(agents)]

generator = lambda : tictactoe_v3.env(render_mode=None)
idqn_id = None # randomly select x or o
gamma = 1
tau = 0.95
decay = ExponentialDecay(0.001)

sim = IDQNEnvironment(generator, idqn_agents, gamma, tau, decay, seed)
episodes = 5000
rh, eh = sim.train(episodes)

path  = f'{Path(__file__).resolve().parent.parent}/res/figures'
agents = {agnt_id : f'idqn_{i}' for i, agnt_id in enumerate(sim.agents.keys())}
width = 100

save_results(agents, rh, eh, f'{path}/v0_idqn_idqn', width)