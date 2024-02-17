import itertools
import numpy as np
import pandas as pd

from ..utils.utils import find_convergence_point
from .eps_decay import LinearDecay
from ..agents.dqn.dqn_agent import DQNAgent
from ..game.dqn_dqn_ttt import TicTacToeDQNDQN
    
if __name__ == "__main__":

    states = 18
    actions = 9
    seed = 42
    episodes = 7500
    decay = LinearDecay(episodes)
    width = 100
    thresh = 0.1

    # hyperparameter search
    columns = ['gamma', 'tau', 'sync', 'batch_size', 'hidden_size']
    gammas = [0.99, 1]
    taus = [0.85, 0.95, 1]
    syncs = [2, 5, 9]
    batch_sizes = [16, 32, 64, 96]
    hidden_sizes = [16, 32, 64]
    cartesian_product = list(itertools.product(gammas, taus, syncs, batch_sizes, hidden_sizes))
    hyperparams = pd.DataFrame(cartesian_product, columns=columns)
    res_col = ['x_cnvg', 'o_cnvg', 'x_cnvg_val', 'o_cnvg_val', 'x_rwd', 'o_rwd']
    res = pd.DataFrame(columns = res_col, index = hyperparams.index)

    for i, (gamma, tau, sync, bs, hs) in hyperparams.iterrows():
        print(f'iteration {i+1}: {gamma, tau, sync, bs, hs}')
        bs, hs = int(bs), int(hs)
        dqnx = DQNAgent(states, actions, hidden_size=hs, seed=seed, sync=sync, batch_size=bs)
        dqno = DQNAgent(states, actions, hidden_size=hs, seed=seed, sync=sync, batch_size=bs)
        ttt = TicTacToeDQNDQN(dqnx, dqno, gamma, tau, decay)
        rh, _ = ttt.train(episodes)
        agents = {ttt.agents[0] : 'x', ttt.agents[1] : 'o'}

        for agent_id, history in rh.items():
            cp = find_convergence_point(history)
            res.loc[i, f'{agents[agent_id]}_cnvg'] = cp
            res.loc[i, f'{agents[agent_id]}_cnvg_val'] = history[cp:].mean() * (cp != -1) -(cp == -1)
            res.loc[i, f'{agents[agent_id]}_rwd'] = history[-width:].mean()

    hyperparams.to_csv('./hyperparameters/linear/hyperparams.csv')
    res.to_csv('./hyperparameters/linear/results.csv')