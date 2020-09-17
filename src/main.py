# uncomment when uploading to kaggle competition to add the location of our
# source files on the kaggle server to the python path
# import sys
# sys.path.append("/kaggle_simulations/agent")

import numpy as np

from bounties import Bounties
from convert import convert
from move import move
from spawns import Spawns
from state import State
from targets import Targets


# stores lists of ships / yards pending decisions, as well as a dictionary
# of ships / yards together with the actions decided for them
class Actions:
    def __init__(self, state):
        self.decided = {}
        self.ships = list(state.my_ships)
        self.yards = list(state.my_yards)
        return

    def asdict(self):
        return {k: v for k, v in self.decided.items() if v is not None}


# global variables that store list of opponent ships we are hunting
# and yards we are protecting between turns
ship_target_memory = []
protection_memory = np.array([], dtype=int)


def agent(obs, config):
    # read (obs, config) into internal game state object
    state = State(obs, config)

    # actions object stores a list of pending ships/yards. as we decide on
    # actions, we remove the ships/yards from the pending lists and store
    # them in a dictionary together with their actions
    actions = Actions(state)

    # convert appropriate ships into yards
    global protection_memory
    protection_memory = convert(state, actions, protection_memory)

    # plan where we want to spawn new ships
    spawns = Spawns(state, actions)

    # place bounties on selected opponent ships/yards
    global ship_target_memory
    bounties = Bounties(state, ship_target_memory)
    ship_target_memory = bounties.target_list

    # set destinations for ships and rank moves by how much closer
    # we get to the destinations
    targets = Targets(state, actions, bounties, spawns, protection_memory)

    # decide on moves for ships
    move(state, actions, targets, protection_memory)

    # spawn the new ships at unoccupied shipyards
    spawns.spawn(state, actions)

    return actions.asdict()
