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
  # WAstar_agent = WeightedAStarAgent()
  # actions, total_cost, expanded = WAstar_agent.search(env, h_weight=0.7)
  # print(f"Total_cost: {total_cost}")
  # print(f"Expanded: {expanded}")
  # print(f"Actions: {actions}")
  # Astar_agent = AStarAgent()
  # actions, total_cost, expanded = Astar_agent.search(env)
  # print(f"Total_cost: {total_cost}")
  # print(f"Expanded: {expanded}")
  # print(f"Actions: {actions}")
  # UCS_agent = UCSAgent()
  # actions, total_cost, expanded = UCS_agent.search(env_4)
  # print(f"Total_cost: {total_cost}")
  # print(f"Expanded: {expanded}")
  # print(f"Actions: {actions}")

  import csv

test_maps = {
    "map12x12": ['SFAFTFFTHHHF',
                 'AFLTFFFFTALF',
                 'LHHLLHHLFTHP',
                 'HALTHAHHAPHF',
                 'FFFTFHFFAHFL',
                 'LLTHFFFAHFAT',
                 'HAAFFALHTATF',
                 'LLLFHFFHTLFH',
                 'FATAFHTTFFAF',
                 'HHFLHALLFTLF',
                 'FFAFFTTAFAAL',
                 'TAAFFFHAFHFG'],
    "map15x15": ['SFTTFFHHHHLFATF',
                 'ALHTLHFTLLFTHHF',
                 'FTTFHHHAHHFAHTF',
                 'LFHTFTALTAAFLLH',
                 'FTFFAFLFFLFHTFF',
                 'LTAFTHFLHTHHLLA',
                 'TFFFAHHFFAHHHFF',
                 'TTFFLFHAHFFTLFP',
                 'TFHLHTFFHAAHFHF',
                 'HHAATLHFFLFFHLH',
                 'FLFHHAALLHLHHAT',
                 'TLHFFLTHFTTFTTF',
                 'AFLTPAFTLHFHFFF',
                 'FFTFHFLTAFLHTLA',
                 'HTFATLTFHLFHFAG'],
    "map20x20" : ['SFFLHFHTALHLFATAHTHT',
                  'HFTTLLAHFTAFAAHHTLFH',
                  'HHTFFFHAFFFFAFFTHHHT',
                  'TTAFHTFHTHHLAHHAALLF',
                  'HLALHFFTHAHHAFFLFHTF',
                  'AFTAFTFLFTTTFTLLTHPF',
                  'LFHFFAAHFLHAHHFHFALA',
                  'AFTFFLTFLFTAFFLTFAHH',
                  'HTTLFTHLTFAFFLAFHFTF',
                  'LLALFHFAHFAALHFTFHTF',
                  'LFFFAAFLFFFFHFLFFAFH',
                  'THHTTFAFLATFATFTHLLL',
                  'HHHAFFFATLLALFAHTHLL',
                  'HLFFFFHFFLAAFTFFPAFH',
                  'HTLFTHFFLTHLHHLHFTFH',
                  'AFTTLHLFFLHTFFAHLAFT',
                  'HAATLHFFFHHHHAFFFHLH',
                  'FHFLLLFHLFFLFTFFHAFL',
                  'LHTFLTLTFATFAFAFHAAF',
                  'FTFFFFFLFTHFTFLTLHFG'],
}

test_envs = {}
for map_name, map_inst in test_maps.items():
    test_envs[map_name] = CampusEnv(map_inst)


DFSG_agent = DFSGAgent()
UCS_agent = UCSAgent()
WAStar_agent = WeightedAStarAgent()
AStar_agent = AStarAgent()

weights = [0.3, 0.7, 0.9]

agents_search_function = [
    DFSG_agent.search,
    UCS_agent.search,
    AStar_agent.search,
]

header = [
    'map',
    'DFS-G cost', 'DFS-G num of expanded nodes',
    'UCS cost', 'UCS  num of expanded nodes',
    'A* cost', 'A*  num of expanded nodes',
    'W-A* (0.3) cost', 'W-A* (0.3) num of expanded nodes',
    'W-A* (0.7) cost', 'W-A* (0.7) num of expanded nodes',
    'W-A* (0.9) cost', 'W-A* (0.9) num of expanded nodes',
]

with open("results.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for env_name, env in test_envs.items():
        data = [env_name]
        for agent in agents_search_function:
            _, total_cost, expanded = agent(env)
            data += [total_cost, expanded]
        for w in weights:
            _, total_cost, expanded = WAStar_agent.search(env, w)
            data += [total_cost, expanded]
        writer.writerow(data)