import os
import sys
import time

from typing import List, Tuple
from CampusEnv import CampusEnv
from Algorithms import *

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

MAPS = {
    "4x4": ["SFFF",
            "FHFH",
            "FFFH",
            "HFFG"],
    "8x8": ["SFFFFFFF",
            "FFFFFTAL",
            "TFFHFFTF",
            "FPFFFHTF",
            "FAFHFPFF",
            "FHHFFFHF",
            "FHTFHFTL",
            "FLFHFFFG"],
}

def print_solution(actions,env: CampusEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
      state, cost, terminated = env.step(action)
      total_cost += cost
    #   clear_output(wait=True)

      print(env.render())
      print(f"Timestep: {i + 2}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Cost: {cost}")
      print(f"Total cost: {total_cost}")

      time.sleep(1)

      if terminated is True:
        break

if __name__ == "__main__":
  env = CampusEnv(MAPS["8x8"])
  env_4 = CampusEnv(MAPS["4x4"])
  state = env.reset()
  env_4.reset()

  # DFSG_agent = DFSGAgent()
  # actions, total_cost, expanded = DFSG_agent.search(env)

  # print(f"Total_cost: {total_cost}")
  # print(f"Expanded: {expanded}")
  # print(f"Actions: {actions}")

  # assert total_cost == 148.0, "Error in total cost returned"
  # assert expanded == 20, "Error in number of expanded nodes returned"
  # assert actions == [0,0,0,0,0,0,0,1,1,2,1,2,1,1,0,0,1,1]

  # UCS_agent = UCSAgent()
  # actions, total_cost, expanded = UCS_agent.search(env)
  WAstar_agent = WeightedAStarAgent()
actions, total_cost, expanded = WAstar_agent.search(env, h_weight=0.7)
  print(f"Total_cost: {total_cost}")
  print(f"Expanded: {expanded}")
  print(f"Actions: {actions}")

  # UCS_agent = UCSAgent()
  # actions, total_cost, expanded = UCS_agent.search(env_4)
  # print(f"Total_cost: {total_cost}")
  # print(f"Expanded: {expanded}")
  # print(f"Actions: {actions}")