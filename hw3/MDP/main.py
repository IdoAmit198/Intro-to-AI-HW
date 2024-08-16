
from mdp_rl_implementation import value_iteration, get_policy, policy_evaluation, policy_iteration, adp_algorithm
from mdp import MDP
from simulator import Simulator

def example_driver():
    """
    This is an example of a driver function, after implementing the functions
    in "mdp_rl_implementation.py" you will be able to run this code with no errors.
    """

    mdp = MDP.load_mdp()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@ The board and rewards @@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    
    mdp.print_rewards()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Value iteration @@@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    U = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    print("\nInitial utility:")
    mdp.print_utility(U)
    print("\nFinal utility:")
    U_new = value_iteration(mdp, U)
    mdp.print_utility(U_new)
    print("\nFinal policy:")
    policy = get_policy(mdp, U_new)
    mdp.print_policy(policy)

    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Policy iteration @@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("\nPolicy evaluation:")
    U_eval = policy_evaluation(mdp, policy)
    mdp.print_utility(U_eval)

    policy = [['UP', 'UP', 'UP', 0],
              ['UP', 'WALL', 'UP', 0],
              ['UP', 'UP', 'UP', 'UP']]

    print("\nInitial policy:")
    mdp.print_policy(policy)
    print("\nFinal policy:")
    policy_new = policy_iteration(mdp, policy)
    mdp.print_policy(policy_new)

    print("Done!")
    


def adp_example_driver():

    sim = Simulator()

    reward_matrix, transition_probabilities = adp_algorithm(sim,num_episodes=10)
  
    print("Reward Matrix:")
    print(reward_matrix)
    
    print("\n Transition Probabilities:")
    for action, probs in transition_probabilities.items():
        print(f"{action}: {probs}")

def adp_learner_driver():
    sim = Simulator()
    for num_episodes in [10, 100, 1000]:
        mdp = MDP.load_mdp()
        # Create a list of actions from dict keys of actions
        actions_list = list(mdp.actions.keys())
        reward_matrix, transition_probabilities = adp_algorithm(sim,num_episodes=num_episodes, num_rows=mdp.num_row, num_cols=mdp.num_col, actions=actions_list)
        board = []
        for row in reward_matrix:
            row_string = [str(cell) if cell != 0 else "WALL" for cell in row]
            board.append(row_string)
        # for cell in reward_matrix.tolist():
        #     string_cell = str(cell) if cell != 0 else "WALL"
        #     board.append(string_cell)
        mdp.board = board
        mdp.transition_function = transition_probabilities
        # mdp = MDP(board= board,
        #           terminal_states=[(0, 3)],
        #           transition_function=transition_probabilities,
        #           gamma=0.9)
        policy = [['UP', 'UP', 'UP', 0],
              ['UP', 'WALL', 'UP', 0],
              ['UP', 'UP', 'UP', 'UP']]
        policy_new = policy_iteration(mdp, policy)
        print(f"For {num_episodes} num episodes:")
        print("Reward Matrix:")
        print(reward_matrix)
        
        print("\n Transition Probabilities:")
        for action, probs in transition_probabilities.items():
            print(f"{action}: {probs}")
        mdp.print_policy(policy_new)
        print("#"*50)
    return reward_matrix, transition_probabilities
    
if __name__ == '__main__':
    # run our example
    # example_driver()
    # adp_example_driver()
    adp_learner_driver()
