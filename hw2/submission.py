import numpy as np
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, Robot, board_size, manhattan_distance
import random

import math
import time


# TODO: section a : 3

def is_holding_package(robot: Robot):
    if robot.package is None:
        return False
    return True

def batteryR(env: WarehouseEnv, robot_id: int, robot_dist_from_pack_credit: int):
    my_robot = env.get_robot(robot_id)
    opponet_robot = env.get_robot((robot_id + 1) % 2)
    if opponet_robot.battery <= 0 and my_robot.credit > opponet_robot.credit:
        return (board_size**2) * my_robot.battery
    if robot_dist_from_pack_credit < my_robot.battery:
        return 0
    dist_to_closest_charge_station =  min([manhattan_distance(my_robot.position, chrage_station.position) \
                                           for chrage_station in env.charge_stations])
    return dist_to_closest_charge_station


##################################################################
def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def has_package(robot):
    return robot.package is not None

def dist(env, robot_id):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    if has_package(robot):
        if robot.position == robot.package.destination:
            return -(board_size ** 3)
        return manhattan_dist(robot.position, robot.package.destination)
    elif has_package(other_robot):
        assert len(env.packages) == 3
        free_package = env.packages[0]
        return manhattan_dist(robot.position, free_package.position) + \
            manhattan_dist(free_package.position, free_package.destination)
    else:
        packages = env.packages
        dist_to_package = dict()
        min_dist = 15
        closest_package = None
        for package in packages:
            my_dist = manhattan_dist(robot.position, package.position)
            if my_dist < min_dist:
                min_dist = my_dist
                closest_package = package
            dist_to_package[package] = (my_dist, manhattan_dist(other_robot.position, package.position))
        if dist_to_package[closest_package][0] < dist_to_package[closest_package][1]:
            return dist_to_package[closest_package][0] + manhattan_dist(closest_package.position,
                                                                        closest_package.destination)
        other_package = None
        for package in packages:
            if package != closest_package:
                other_package = package
                break
        return dist_to_package[other_package][0] + manhattan_dist(other_package.position, other_package.destination)


########################################################################


def stepsR(env: WarehouseEnv, robot_id: int, T: int):
    my_robot = env.get_robot(robot_id)
    opponet_robot = env.get_robot((robot_id - 1) % 2)
    if is_holding_package(my_robot) and my_robot.package.destination != my_robot.position:
        # The robot is holding a package and it is not at the destination
        return manhattan_distance(my_robot.package.destination, my_robot.position)
    
    elif is_holding_package(my_robot) and my_robot.position == my_robot.package.destination:
        # The robot is holding a package and it is at the destination
        return -T
    
    elif is_holding_package(opponet_robot):
        # My robot does not hold a package and the opponent robot does
        package_on_board = env.packages[0]
        robot_to_package_dist = manhattan_distance(my_robot.position, package_on_board.position)
        package_orig_dest_dist = manhattan_distance(package_on_board.destination, package_on_board.position)
        return robot_to_package_dist + package_orig_dest_dist
            
    else:
        # Neither of the robots are holding a package
        closest_package = None
        min_dist_from_package = 4 * (board_size-1)
        # packages_dist_from_me = []
        # packages_dist_from_opponent = []
        for package in env.packages[:2]:
            opponet_dist_from_package = manhattan_distance(package.position, opponet_robot.position)
            my_dist_from_package = manhattan_distance(package.position, my_robot.position)
            if my_dist_from_package < min_dist_from_package and my_dist_from_package < opponet_dist_from_package:
                min_dist_from_package = my_dist_from_package
                closest_package = package
        if not closest_package:
            package_0_dist = manhattan_distance(env.packages[0].position, my_robot.position)
            package_1_dist = manhattan_distance(env.packages[1].position, my_robot.position)
            if package_0_dist > package_1_dist:
                closest_package = env.packages[0]
                min_dist_from_package = package_0_dist
            else:
                closest_package = env.packages[1]
                min_dist_from_package = package_1_dist
        return min_dist_from_package + manhattan_distance(closest_package.destination, closest_package.position)

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    credit_weight = 5
    # T = 7 * board_size**2
    T = 4
    my_robot = env.get_robot(robot_id)
    opponet_robot = env.get_robot((robot_id + 1) % 2)
    my_robot_dist_from_package_credit = stepsR(env=env, robot_id=robot_id, T=T)
    # d = dist(env, robot_id)
    heuristic =  credit_weight * (my_robot.credit - opponet_robot.credit) \
                - batteryR(env, robot_id, my_robot_dist_from_package_credit) \
                - my_robot_dist_from_package_credit 
                # - batteryR(env, robot_id, d) -d
    return heuristic



class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id: int, time_limit):
        self.time_limit= time.time()+0.9*time_limit
        return self.minimax(env,agent_id)
    
    #main function
    def minimax(self, env: WarehouseEnv, agent_id: int):
        D = 2
        result_operator, result_heuristic = None, -math.inf
        while time.time() < self.time_limit:
            curr_operator, curr_heuristic = self.RB_minimax(env,agent_id,D)
            if curr_heuristic > result_heuristic:
                result_operator = curr_operator
                result_heuristic = curr_heuristic
            D = D + 2
        return result_operator

    #for loop
    def RB_minimax(self, env: WarehouseEnv, agent_id: int, depth):
        operators, children = self.successors(env, agent_id)
        childs_heuristic = [self.min_func(child, agent_id + 1, depth - 1) for child in children]
        # print(f"for depth {depth}:\n")
        # for op, res in zip(operators, childs_heuristic):
        #     print(f"op: {op}, got heuristic: {res}")
        max_heuristic_child, max_heuristic_idx = max(childs_heuristic), int(np.argmax(childs_heuristic))
        return operators[max_heuristic_idx], max_heuristic_child
    
    #if it is the agent turn
    def max_func(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) or depth==0 or env.done():
            return smart_heuristic(env,agent_id)
        _, children = self.successors(env, agent_id)
        children_vector = [self.min_func(child, agent_id + 1, depth - 1) for child in children]
        return max(children_vector)
        
    #if it is the other agent turn
    def min_func(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) or depth==0 or env.done():
            return smart_heuristic(env,agent_id)
        _, children = self.successors(env, agent_id)
        children_vector = [self.max_func(child, agent_id + 1, depth - 1) for child in children]
        return min(children_vector)


class AgentAlphaBeta(Agent):
    # section c : 1 - done
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # TODO: Refactor the time_limit computation.
        self.time_limit = time.time() + 0.9 * time_limit
        return self.id_alpha_beta(env, agent_id)

    def max_func(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        agent_id = agent_id % 2
        if time.time() > self.time_limit or depth == 0:
            return self.heuristic(env, agent_id)
        curr_max = -math.inf
        _, children = self.successors(env, agent_id)
        for child in children:
            # child.apply_operator(agent_id, operator)
            curr_result = self.min_func(child, agent_id + 1, depth - 1, alpha, beta)
            curr_max = max(curr_result, curr_max)
            if curr_max >= beta:
                return math.inf
            alpha = max(curr_max, alpha)
        return curr_max
    
    def min_func(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        agent_id = agent_id % 2
        if time.time() > self.time_limit or depth == 0:
            return smart_heuristic(env, agent_id)
        curr_min = math.inf
        _, children = self.successors(env, agent_id)
        for child in children:
            # child.apply_operator(agent_id, operator)
            curr_result = self.max_func(child, agent_id + 1, depth - 1, alpha, beta)
            curr_min = min(curr_result, curr_min)
            if curr_min <= alpha:
                return -math.inf
            beta = min(curr_min, beta)
        return curr_min


    def id_alpha_beta(self, env: WarehouseEnv, agent_id: int):
        D = 2
        result_operator, result_heuristic = None, -math.inf
        while time.time() < self.time_limit:
            curr_operator, curr_heuristic = self.alpha_beta(env, agent_id, D, -math.inf, math.inf)
            if curr_heuristic > result_heuristic:
                result_operator = curr_operator
                result_heuristic = curr_heuristic
            D = D + 2
        return result_operator
        # while time.time() < self.time_limit:
        #     res = self.alpha_beta(env, agent_id, D, -math.inf, math.inf)
        #     D = D + 2
        # return res

    def alpha_beta(self, env: WarehouseEnv, agent_id: int, depth, alpha, beta):
        operators, children = self.successors(env, agent_id)
        childs_heuristic = [self.min_func(child, agent_id + 1, depth - 1, alpha, beta) for child in children]
        # max_children_idx = int(np.argmax(children_results))
        # index_selected = children_results.index(max_children)
        max_heuristic_child, max_heuristic_idx = max(childs_heuristic), int(np.argmax(childs_heuristic))
        return operators[max_heuristic_idx], max_heuristic_child


class AgentExpectimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.time_limit= time.time()+0.9*time_limit
        return self.expectimax(env,agent_id)

    #main function
    def expectimax(self, env: WarehouseEnv, agent_id: int):
        D = 2
        self.agent_id = agent_id
        chosen_operator, result_max_val = None, -math.inf
        while time.time() < self.time_limit:
            operators, children = self.successors(env, agent_id)
            # max_val = None
            # chosen_op = -1
            # for child in children:
            children_vals = [self.RB_expectimax(child,agent_id + 1,D-1) for child in children]
            max_val_child, max_val_idx = max(children_vals), int(np.argmax(children_vals))
            if max_val_child > result_max_val:
                result_max_val = max_val_child
                chosen_operator = operators[max_val_idx]
            D = D + 2
        return chosen_operator

    #inner expectimax
    def RB_expectimax(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) or depth==0 or env.done(): 
            return smart_heuristic(env, agent_id)
        if self.agent_id == agent_id:
            return self.max_func(env, agent_id, depth)
        else: 
            return self.probablistic_func(env, agent_id, depth)
    

    #if it is the agent turn
    def max_func(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) or depth==0 or env.done(): return smart_heuristic(env,agent_id)
        _, children = self.successors(env, agent_id)
        cur_max = -math.inf
        children_vector = [self.RB_expectimax(child, agent_id + 1, depth - 1) for child in children]
        return max(max(children_vector),cur_max)
        
    #if it is the other agent turn and min part
    def min_func(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) or depth==0 or env.done(): return smart_heuristic(env,agent_id)
        _, children = self.successors(env, agent_id)
        cur_min = math.inf
        children_vector = [self.RB_expectimax(child, agent_id + 1, depth - 1) for child in children]
        return min(min(children_vector),cur_min)
    
    #if we need to do expectimax on the results of the other agent
    def probablistic_func(self, env: WarehouseEnv, agent_id: int, depth):
        agent_id = agent_id % 2
        if (time.time() > self.time_limit) or depth==0 or env.done(): return smart_heuristic(env,agent_id)
        operators, children = self.successors(env, agent_id)
        operators_prob = []
        for operator in operators:
            if operator == 'move east' or operator == 'pick up':
                operators_prob.append(2)
            else:
                operators_prob.append(1)
        
        denominator = sum(operators_prob)
        operators_prob = [op / denominator for op in operators_prob]
        heuristics = [self.min_func(child, agent_id + 1, depth - 1) for child in children]
        heuristics = [p * op for p,op in zip(operators_prob,heuristics)]
        return sum(heuristics)
        

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)