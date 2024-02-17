import matplotlib.pyplot as plt
import numpy as np
import torch

from ..dqn.agents.idqn import IDQNAgent

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
    
def find_convergence_point(arr, tolerance=0.05):
    """
    Finds the first index in an array where the absolute differences between subsequent elements
    converge within a specified tolerance and continue to do so for all subsequent elements.

    Parameters:
    - arr (array-like): The input array to check for convergence. Expected to be a NumPy array or a list.
    - tolerance (float, optional): The tolerance within which the absolute differences between
      subsequent elements must fall to consider them as converging. Defaults to 0.05.

    Returns:
    - int: The index of the first element in the array where convergence within the specified tolerance
      is achieved and sustained for all subsequent elements. Returns -1 if no such convergence point is found.

    """
    import numpy as np

    diffs = np.abs(np.diff(arr))
    within_tolerance = diffs <= tolerance
    all_subsequent_within_tol = np.cumprod(within_tolerance[::-1])[::-1]
    first_converge_index = np.argmax(all_subsequent_within_tol)
    if all_subsequent_within_tol[first_converge_index] == 1:
        return first_converge_index
    return -1


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
    indices = np.array(list(map(sum, zip(*reward_history.values())))) != 0
    illegal_percent = {agent_id : rolling(rh * indices, f_illegal, width)
                       for agent_id, rh in reward_history.items()}
    
    plt.subplot(222)
    for agent_id, rew in illegal_percent.items():
        plt.plot(rew, label=agents[agent_id])
    plt.legend()
    plt.ylim(0, 1)
    plt.title(f'rolling width {width} % illegal action')

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

    plt.savefig(f'{plot_name}.png')
    plt.close()

def save_policy(agent: IDQNAgent, policy_name: str):
    """
    Saves the policy of a given DQNAgent to disk.

    Parameters:
    - agent: The IDQNAgent whose policy is to be saved.
    - policy_name: The name under which to save the policy.
    """
    policy = agent.policy.state_dict()
    torch.save(policy, f"{policy_name}.pt")