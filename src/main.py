# uncomment when uploading to kaggle competition to add the location of our
# source files on the kaggle server to the python path
# import sys
# sys.path.append("/kaggle_simulations/agent")

# import source files
from bounties import Bounties
from convert import convert
from move import move
from spawns import Spawns
from state import State
from targets import Targets
# from memory import Memory  # REMOVE


# object to store ships / yards pending decisions as well as
# actions that we have decided on
class Actions:
    def __init__(self, state):
        self.decided = {}
        self.ships = list(state.my_ships)
        self.yards = list(state.my_yards)
        return

    def asdict(self):
        return {k: v for k, v in self.decided.items() if v is not None}


# REMOVE
# memory = Memory()


ship_target_memory = []


def agent(obs, config):
    # read (obs, config) into internal game state object
    state = State(obs, config)

    # REMOVE
    # memory.statistics(state)

    # actions object stores a list of pending ships/yards. as we decide on
    # actions, we remove the ships/yards from the pending lists and store
    # them in a dictionary together with their actions
    actions = Actions(state)

    # convert appropriate ships into yards
    convert(state, actions)

    # plan where we want to spawn new ships
    spawns = Spawns(state, actions)

    # place bounties on selected opponent ships/yards and remember
    # which ships we set bounties on for the future
    global ship_target_memory
    bounties = Bounties(state, ship_target_memory)
    ship_target_memory = bounties.target_list

    # set destinations for ships and rank moves by how much closer
    # we get to the destinations
    targets = Targets(state, actions, bounties, spawns)

    # decide on moves for ships
    move(state, actions, targets)

    # spawn the new ships at unoccupied shipyards
    spawns.spawn(state, actions)

    # REMOVE
    # memory.ship_targets = ship_target_memory.copy()
    # memory.total_bounties += bounties.new_targets_set
    # if 2 + state.step == state.total_steps:
    #     memory.summary()

    return actions.asdict()
