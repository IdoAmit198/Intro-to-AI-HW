import numpy as np
from collections import deque

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self,parent,action_creatiation, cell , cost, heuristic, total_cost):
        self.parent = parent
        self.action_creatiation = action_creatiation
        self.cell = cell
        self.children = []
        self.cost = cost
        self.heuristic = heuristic
        # self.total_cost = np.inf
        self.total_cost = total_cost

    def expend(self,env, Agent_search_type:str = None):
        if self.total_cost == np.inf:
            return []
        children = []
        node_state = env.get_state()
        for action in range(4):
            new_state, cost, terminated = env.step(action)
            # if Agent_search_type == str(DFSGAgent) and cost != np.inf:
            #     cost=1
            child = Node(self,action,new_state,cost,None,self.total_cost+cost)
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
        # self.open.append(self.start_node)
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
                    actions_path, total_cost = self.solution(node)
                    return actions_path, total_cost, self.expanded
                children = node.expend(self.env, str(type(self)))
                self.expanded += 1
                if len(children)>0:
                    for child in children:
                        if self.closed.get(child.cell) is None and self.open.get(child.cell) is None: # self.open keep Node objects, might be wrong to check it, and better to check for integers of cells.
                            self.open[child.cell] = (child.cost, child.cell, child)
                        elif self.open.get(child.cell) and self.open.get(child.cell)[0]>child.cost:
                            self.open[child.cell] = (child.cost, child.cell, child)

            except KeyError:
                print("No solution found.")
                return None

             
    





class WeightedAStarAgent(Agent):
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError   



class AStarAgent(Agent):
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

