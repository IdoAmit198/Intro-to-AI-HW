# Segel imports
from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np

# Our imports
from copy import deepcopy


def Util_calc(mdp: MDP, U: np.ndarray, start_row: int, start_col: int, desired_action: Action) -> float:
    expected_utility = 0.0
    for index, actual_action in enumerate(mdp.actions.keys()):
        prob = mdp.transition_function[desired_action][index]
        step_row, step_col = mdp.step((start_row, start_col), actual_action)
        util = U[step_row][step_col]
        expected_utility += prob * util
    return expected_utility

def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    
    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    U_tag = deepcopy(U_init)
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if U_tag[row][col] != "WALL":
                U_tag[row][col] = float(U_tag[row][col])

    while True:
        U = deepcopy(U_tag)
        delta = 0
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if mdp.board[row][col] == "WALL":
                    continue
                elif (row, col) in mdp.terminal_states:
                    U_tag[row][col] = float(mdp.get_reward((row, col)))
                    continue
                max_val = -np.inf
                for desired_action in mdp.actions.keys():
                    reward = float(mdp.get_reward((row, col)))
                    expected_utility = reward + mdp.gamma * Util_calc(mdp, U, row, col, desired_action)
                    max_val = max(max_val, expected_utility)
                U_tag[row][col] = max_val
                delta = max(delta, np.abs(U[row][col] - U_tag[row][col]))
        if delta < (epsilon*(1-mdp.gamma))/mdp.gamma:
            break
    U_final = U_tag
    # ========================
    return U_final

def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    
    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    policy = deepcopy(U)
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if (row, col) in mdp.terminal_states or mdp.board[row][col] == "WALL":
                policy[row][col] = None
                continue
            util_list = [Util_calc(mdp, U, row, col, action) for action in mdp.actions]
            util_list = np.array(util_list)
            util_list = float(mdp.get_reward((row,col))) + mdp.gamma*util_list
            action_list = [action for action in mdp.actions]
            max_index = np.argmax(util_list)
            policy[row][col] = action_list[max_index]
    # ========================
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # TODO:
    # ====== YOUR CODE: =====
    """"
    We will follow the pseudo code from the tutorial and construct the linear equation.
    Then, we will solve the linear equation using numpy's linalg.solve function:
    U = R + gammaP*U  ->  U = inv(I - gammaP) * R
    U will be the utilities vector under the given policy, which is our solution.
    R will be the rewards vector, which is the constant vector in the linear equation.
    P will be the transition matrix, which is the coefficients matrix in the linear equation.
    Finally, we will reshape U to be a 2D array with the same shape as the board.
    """
    R = [] # Vector of rewards
    P = np.zeros((mdp.num_row * mdp.num_col, mdp.num_row * mdp.num_col)) # Matrix of transition probabilities
    for row,col in np.ndindex(np.array(mdp.board).shape):
        if mdp.board[row][col] == "WALL":
            R.append(0)
            continue
        elif (row, col) in mdp.terminal_states:
            R.append(float(mdp.get_reward((row, col))))
            continue
        reward = float(mdp.get_reward((row, col)))
        R.append(reward) # This way we can keep the order of the rewards corresponding to the order of the states
        action = policy[row][col]
        for index, action in enumerate(mdp.actions.keys()):
            step_row, step_col = mdp.step((row, col), action)
            P[row * mdp.num_col + col, step_row * mdp.num_col + step_col] += \
                mdp.transition_function[Action(policy[row][col])][index]
    R = np.array(R)
    U = np.linalg.solve(np.eye(mdp.num_row * mdp.num_col) - mdp.gamma * P, R)
    U = U.reshape(np.array(mdp.board).shape)
    return U
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    policy = deepcopy(policy_init)
    while True:
        U = policy_evaluation(mdp, policy)
        unchanged = True
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                if (row,col) in mdp.terminal_states or mdp.board[row][col] == "WALL":
                    policy[row][col] = None
                    continue
                util_list = [Util_calc(mdp,U,row,col,action) for action in mdp.actions]
                action_list = [action for action in mdp.actions]
                max_index = np.argmax(util_list)
                max_action = action_list[max_index]
                if max_action != policy[row][col]:
                    policy[row][col] = max_action
                    unchanged = False
        if unchanged:
            break
    optimal_policy = policy
    # ========================
    return optimal_policy


def adp_algorithm(
    sim: Simulator, 
    num_episodes: int,
    num_rows: int = 3, 
    num_cols: int = 4, 
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] 
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """
    

    transition_probs = None
    reward_matrix = None
    # TODO
    # ====== YOUR CODE: ======
    # Initialize the reward matrix and the transition probabilities dictionary
    reward_matrix = np.zeros((num_rows, num_cols))
    transition_probs = {action: [0]*len(actions) for action in actions}
    # Run the simulation
    for episode_index, episode_gen in enumerate(sim.replay(num_episodes=num_episodes)):
        for step in episode_gen:
            state, reward, action, actual_action = step
            reward_matrix[state[0], state[1]] = reward
            # If either of the actions is None, we reached a terminal state and only need the reward.
            if actual_action is not None:
            # Get the indices of the actual_action in the actions list
                actual_action_index = actions.index(actual_action) if actual_action is not None else None
                transition_probs[action][actual_action_index] += 1
    # Normalize the transition probabilities
    for action in actions:
        total = sum(transition_probs[action])
        for next_idx, next_action in enumerate(actions):
            transition_probs[action][next_idx] /= total
    # ========================
    return reward_matrix, transition_probs 
