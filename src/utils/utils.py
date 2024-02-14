import matplotlib.pyplot as plt
import numpy as np
import torch

from ..agents.dqn_agent import DQNAgent

def rolling(arr, f, width=100):
    """
    Applies a function over a rolling window of the given array.

    Parameters:
    - arr: The input array to process.
    - f: The function to apply over the rolling window. This function should accept the windowed array and any additional fixed arguments, and return a single value.
    - width: The width of the rolling window.

    Returns:
    - An array of values obtained by applying the function `f` over the rolling window of `arr`.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    windowed_arr = sliding_window_view(arr, window_shape=width)
    return f(windowed_arr)
    
def save_results(agents, reward_history, epsilon_history, plot_name, width=100):
    """
    Saves a set of plots illustrating various statistics of agents' performance.

    Parameters:
    - agents: Dictionary mapping agent IDs to agent names.
    - reward_history: Dictionary mapping agent IDs to their rewards over time.
    - epsilon_history: Array of epsilon values over time.
    - plot_name: Name of the file to save the plot as.
    - width: The width of the rolling window to use for smoothing plots.
    """
    plt.figure(figsize=(16, 10))
        
    # Rolling mean rewards
    f_mean = lambda arr : np.mean(arr, axis=-1)
    rolling_rew = {agent_id: rolling(rh, f_mean, width) for agent_id, rh in reward_history.items()}
    plt.subplot(221)
    for agent_id, rew in rolling_rew.items():
        plt.plot(rew, label=agents[agent_id])
    plt.legend()
    plt.title(f'rolling {width} mean reward')

    # Illegal action %
    f_illegal = lambda arr: np.sum(arr != 0, axis=-1) / np.shape(arr)[-1]
    illegal_percent = rolling(np.array(list(map(sum, zip(*reward_history.values())))),
                              f_illegal,
                              width)
    
    plt.subplot(222)
    plt.plot(illegal_percent)
    plt.title(f'rolling {width} % illegal action')

    # Win rate
    f_winrate = lambda arr: np.sum(arr == 1, axis=-1) / np.shape(arr)[-1]
    winrate = {agent_id: rolling(rh, f_winrate, width) for agent_id, rh in reward_history.items()}
    
    plt.subplot(223)
    for agent_id, rew in winrate.items():
        plt.plot(rew, label=agents[agent_id])
    plt.legend()
    plt.title(f'rolling {width} win rate')

    # Epsilon decay
    plt.subplot(224)
    plt.plot(epsilon_history)
    plt.title('epsilon values')

    plt.savefig(f'./plots/{plot_name}.png')
    plt.close()

def save_policy(agent: DQNAgent, policy_name: str):
    """
    Saves the policy of a given DQNAgent to disk.

    Parameters:
    - agent: The DQNAgent whose policy is to be saved.
    - policy_name: The name under which to save the policy.
    """
    policy = agent.qnetwork_policy.state_dict()
    torch.save(policy, f"./models/{policy_name}.pt")

def state_to_input(observation: np.array) -> torch.tensor:
    """
    Convert a Tic Tac Toe game (3, 3, 2) state to a (18, 1) PyTorch tensor suitable for DQN input.

    Parameters:
    - observation: Numpy array representing the game state.

    Returns:
    - A PyTorch tensor of the flattened game state.
    """
    flattened = observation.flatten()
    return torch.tensor(flattened).float()

def ascii_state(state: torch.tensor):
    """
    Converts a game state tensor back into a human-readable ASCII board.

    Parameters:
    - state: A PyTorch tensor representing the game state.

    Returns:
    - A 3x3 numpy array of strings representing the game board.
    """
    s = state.numpy().reshape(3, 3, 2)
    board = [['_ ' for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(2):
                if k == 0 and s[i][j][k] == 1:
                    board[i][j] = 'x '
                elif k == 1 and s[i][j][k] == 1:
                    board[i][j] = 'o '

    return np.array(board, dtype=str)
