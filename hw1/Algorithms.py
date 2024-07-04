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
        self.total_cost = np.inf

    def expend(self,env):
        children = []
        for action in range(4):
            new_state, cost, terminated = env.step(action)
            if type(self)== DFSGAgent:
                cost=1
            child = Node(self,action,new_state,cost,None,self.total_cost+cost)
            children.append(child)
        return children
    

class Agent:
    def solution(self,node:Node) -> Tuple[List[int], float]:
        total_cost= node.total_cost
        actions_path = []
        while node.parent:
            actions_path.append(node.action_creatiation)
            node = node.parent
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



    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        node = self.open.pop()
        self.closed[node.cell]=node.cell
        if env.is_final_state(node.cell):
            return self.solution(node),self.expanded
        
        for child in node.expend(env):
            if self.closed.get(child.cell) and child not in self.open:
                self.open.append(child)
                result = self.search(env)
                return 1,1,1

    
        

class UCSAgent(Agent):
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
    





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

