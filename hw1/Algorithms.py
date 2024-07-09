import numpy as np
from collections import deque

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self,parent,action_creatiation, cell , cost, heuristic, total_cost, f_value = 0):
        self.parent = parent
        self.action_creatiation = action_creatiation
        self.cell = cell
        self.children = []
        self.cost = cost
        self.heuristic = heuristic
        # self.total_cost = np.inf
        self.total_cost = total_cost
        self.f_value = self.heuristic if f_value is None else f_value

    def expend(self,env, Agent_search_type:str = None, h_weight=0):
        if self.total_cost == np.inf:
            return []
        children = []
        node_state = env.get_state()
        for action in range(4):
            new_state, cost, terminated = env.step(action)
            # if Agent_search_type == str(DFSGAgent) and cost != np.inf:
            #     cost=1
            new_state_huristic = HCampus_huristic(new_state, env)
            f_value = (1-h_weight)*(self.total_cost + cost) + h_weight*new_state_huristic
            child = Node(parent = self,
                         action_creatiation = action,
                         cell = new_state,
                         cost = cost,
                         heuristic = HCampus_huristic(new_state, env),
                         total_cost = self.total_cost+cost,
                         f_value = f_value)
            children.append(child)
            env.set_state(node_state)
            # env.step((action+2)%4)
        return children
    

class Agent:
    def solution(self,node:Node) -> Tuple[List[int], float]:
        total_cost= node.total_cost
        actions_path = []
        while node.parent:
            actions_path.append(node.action_creatiation)
            node = node.parent
        actions_path = actions_path[::-1]
        return actions_path, total_cost

class DFSGAgent(Agent):
    def __init__(self) -> None:
        self.closed={}
        self.open = deque()
        self.start_node = Node(None,None,0,1,None,0)
        # self.start_node = Node(0,0,0,1,0,0)
        self.expanded = 0
        self.open.append(self.start_node)
        super().__init__()

    def recursive_search(self) -> Tuple[List[int], float, int]:
        node = self.open.pop()
        if node.cell is not None:
            self.env.set_state(node.cell)
        else:
            print(f"node.cell is None.")
        self.closed[node.cell]=node.cell
        if self.env.is_final_state(node.cell):
            actions_path, total_cost = self.solution(node)
            return actions_path, total_cost, self.expanded
        children = node.expend(self.env, str(type(self)))
        self.expanded += 1
        if len(children)>0:
            for child in children:
                if not self.closed.get(child.cell) and child not in self.open: # self.open keep Node objects, might be wrong to check it, and better to check for integers of cells.
                    self.open.append(child)
                    result = self.recursive_search()
                    if result:
                        return result
        else:
            return None

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        return self.recursive_search()
     

class UCSAgent(Agent):
  
    def __init__(self) -> None:
        self.closed={}
        self.open = heapdict.heapdict()
        self.start_node = Node(None,None,0,1,None,0)
        # self.start_node = Node(0,0,0,1,0,0)
        self.expanded = 0
        self.expanded_list = []
        self.open[self.start_node.cell] = (self.start_node.cost, self.start_node.cell, self.start_node) # Each element is a tuple of (cost, state)
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        while True:
            try:
                node = self.open.popitem()[1][2]
                self.closed[node.cell]=node.cell
                if node.cell is not None:
                    self.env.set_state(node.cell)
                else:
                    print(f"node.cell is None.")
                if self.env.is_final_state(node.cell):
                    print(self.expanded_list)
                    actions_path, total_cost = self.solution(node)
                    return actions_path, total_cost, self.expanded
                children = node.expend(self.env, str(type(self)))
                self.expanded += 1
                self.expanded_list.append(node.cell)
                if len(children)>0:
                    for child in children:
                        if self.closed.get(child.cell) is None and self.open.get(child.cell) is None: # self.open keep Node objects, might be wrong to check it, and better to check for integers of cells.
                            self.open[child.cell] = (child.total_cost, child.cell, child)
                        elif self.open.get(child.cell) and self.open.get(child.cell)[0]>child.total_cost:
                            self.open[child.cell] = (child.total_cost, child.cell, child)
            except KeyError:
                print("No solution found.")
                return None

def HCampus_huristic(state:int, env:CampusEnv) -> float:
    row, col = env.to_row_col(state)
    goals_list = env.get_goal_states()
    goals_rows_cols = [env.to_row_col(goal) for goal in goals_list]
    # Compute manhatan distance from the state to each of the goals
    manhatan_dist_list = [abs(goal_row-row)+abs(goal_col-col) for goal_row, goal_col in goals_rows_cols]
    min_manhatan = min(manhatan_dist_list)
    return min(min_manhatan, 100)

class WeightedAStarAgent(Agent):
    def __init__(self):
        self.closed={}
        self.open = heapdict.heapdict()
        self.expanded = 0
        self.expanded_list = []
        super().__init__()

    
    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.start_node = Node(None,None,0,1,HCampus_huristic(state=0, env=env),0, HCampus_huristic(state=0, env=env))   
        self.open[self.start_node.cell] = (self.start_node.f_value, self.start_node.cell, self.start_node) # Each element is a tuple of (cost, state)
        while True:
            try:
                node = self.open.popitem()[1][2]
                self.closed[node.cell]=node.total_cost
                if node.cell is not None:
                    self.env.set_state(node.cell)
                else:
                    print(f"node.cell is None.")
                if self.env.is_final_state(node.cell):
                    print(self.expanded_list)
                    actions_path, total_cost = self.solution(node)
                    return actions_path, total_cost, self.expanded
                children = node.expend(self.env, str(type(self)), h_weight=h_weight)
                self.expanded += 1
                self.expanded_list.append(node.cell)
                if len(children)>0:
                    for child in children:
                        if self.closed.get(child.cell) is None and self.open.get(child.cell) is None:
                            self.open[child.cell] = (child.f_value, child.cell, child)
                        elif self.open.get(child.cell) and self.open.get(child.cell)[2].total_cost>child.total_cost:
                            self.open[child.cell] = (child.f_value, child.cell, child)
                        elif self.closed.get(child.cell) and self.closed.get(child.cell)>child.total_cost:
                            self.closed.pop(child.cell)
                            self.open[child.cell] = (child.f_value, child.cell, child)
            except IndexError:
                print("Heap was empty. No solution found.")
                return None


class AStarAgent():
    
    def __init__(self):
        self.weighted_A_star = WeightedAStarAgent()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return self.weighted_A_star.search(env=env, h_weight=0.5)

