# ------------ CODE IMPORTED FROM leading.py ------------ #
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import OPTICS
from uuid import uuid4
from time import time
from copy import deepcopy
import warnings

# parameters for graph weighting
GRAPH_MY_WEIGHT = 2
GRAPH_OPP_WEIGHT = 4

# parameters for yard clustering algorithm
MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 6
MAX_YARDS_PER_SHIP = 1/6

# how many steps are the "initial" and "final" phases of the game
STEPS_INITIAL = 25
STEPS_FINAL = 25
STEPS_INTEREST_SPIKE = 20

HUNTERS_PER_TARGET = 5
MAX_HUNTING_QUANTILE = 1

USE_FIFO = True


# ------------ CODE IMPORTED FROM queue.py ------------ #
class Queue:
    def __init__(self, initkeys=[]):
        self.pending = initkeys
        return

    def not_empty(self):
        return len(self.pending) > 0

    def remove(self, actor):
        self.pending.remove(actor)
        return

    def schedule(self, state):
        # separate actors into ships and yards
        # PUT IN INIT...
        ships = [actor for actor in self.pending if actor in state.my_ships]
        yards = [actor for actor in self.pending if actor in state.my_yards]

        # first schedule any ships that are likely to convert
        converts = [ship for ship in ships if should_convert(ship, state)]
        if len(converts) > 0:
            nextup = converts[0]

        # if there are ships, schedule the ship with the highest priority
        # ships have highest priority if they only have one non-colliding move
        # and ties are broken by amount of cargo (unless some ship has > 1000
        # in cargo, in which case it should be scheduled early anyway)
        elif len(ships) > 0:
            def priority(ship):
                hal = state.my_ships[ship][1]

                legal = state.legal_actions(ship)
                if "CONVERT" in legal:
                    legal.remove("CONVERT")
                no_col = [move for move in legal if
                          (not state.opp_collision(ship, move, strict=True))
                          and (not state.self_collision(ship, move))]

                return 1000 * (len(no_col) <= 1) + hal

            nextup = max(ships, key=priority)

        # else we have only yards left. take yards with small clusters first
        # so ships will spawn where there is the most space
        else:
            clust_size = lambda x: state.get_cluster(state.my_yards[x]).size
            nextup = min(yards, key=clust_size)

        # remove the scheduled actor from the pending list and return it
        self.remove(nextup)
        return nextup


# ------------ CODE IMPORTED FROM survival.py ------------ #
def survival(state, queue):
    actions = {}

    if len(state.my_yards) == 0:
        # see if it is possible to convert the ship with the most cargo
        # if so, convert it, update state, and return to normal operation
        halite = lambda x: state.my_ships[x][1]
        max_cargo = max(queue.pending, key=halite)
        pos, hal = state.my_ships[max_cargo]
        if hal + state.my_halite >= state.config.convertCost:
            queue.remove(max_cargo)
            actions[max_cargo] = "CONVERT"
            state.update(max_cargo, "CONVERT")
        # otherwise instruct all ships to survive as long as possible while
        # collecting as much halite as possible
        else:
            actions = run_and_collect(state, queue)

    # do elif since it can never happen that we start the turn without any
    # ships or yards - but it can happen that the previous block converted
    # our last ship...
    elif len(state.my_ships) == 0:
        # spawn as many ships as we can afford
        actions = spawn_maximum(state, queue)

    return actions


def run_and_collect(state, queue):
    actions = {}

    while queue.not_empty():
        # schedule the next ship

        halite = lambda x: state.my_ships[x][1]
        ship = max(queue.pending, key=halite)
        queue.remove(ship)

        pos, hal = state.my_ships[ship]

        # we want moves that are guaranteed not to collide with opponents
        # and don't collide with our own ships
        legal = state.legal_actions(ship)
        no_self_col = [action for action in legal
                       if not state.self_collision(ship, action)]
        candidates = [action for action in no_self_col
                      if not state.opp_collision(ship, action)]

        # is no such moves exists, revert to not colliding with our own ships
        if len(candidates) == 0:
            candidates = no_self_col

        # if this is still not possible, revert to legal moves
        if len(candidates) == 0:
            candidates = legal

        # we need halite ASAP so try to collect it if there is some
        # this can probably be improved...
        if None in candidates and state.halite_map[pos] > 0:
            action = None
        # otherwise go to site with most halite
        else:
            hal_after = lambda x: state.halite_map[state.newpos(pos, x)]
            action = max(candidates, key=hal_after)

        # update game state with consequence of action
        state.update(ship, action)

        # write action into dictionary of actions to return
        if action is not None:
            actions[ship] = action

    return actions


def spawn_maximum(state, queue):
    actions = {}

    while queue.not_empty():
        # schedule the next yard
        yard = queue.schedule(state)

        # list of legal moves
        legal = state.legal_actions(yard)

        # if spawning is legal, do it
        action = "SPAWN" if "SPAWN" in legal else None

        # update game state with consequence of action
        state.update(yard, action)

        # write action into dictionary of actions to return
        if action is not None:
            actions[yard] = action

    return actions


# ------------ CODE IMPORTED FROM bounties.py ------------ #
class Bounties:
    def __init__(self):
        self.targets = []
        self.hunters = []
        self.targets_pos = np.array([]).astype(int)
        self.targets_hal = np.array([]).astype(int)
        self.rewards = np.array([]).astype(int)
        self.conversions = 0
        self.total_bounties = 0
        self.total_loot = 0
        return

    def not_empty(self):
        return len(self.targets) > 0

    def update(self, state):
        # decide which of our ships should hunt depending on their cargo
        # and how much halite is on the map
        if state.my_ship_pos.size != 0:
            q = MAX_HUNTING_QUANTILE \
                - np.sum(state.halite_map) / state.config.startingHalite
            q = min(1, max(0, q))  # quantiles should be in [0,1]
            cutoff = np.quantile(state.my_ship_hal, q)
            hal = lambda x: state.my_ships[x][1]
            self.hunters = [ship for ship in state.my_ships.keys()
                            if hal(ship) < cutoff]
        else:
            self.hunters = []

        # remove any targets which are no longer eligible
        self.targets = [target for target in self.targets
                        if self.eligible(target, state)]

        # determine how many targets we should have
        n_targets = len(self.hunters) // HUNTERS_PER_TARGET

        # create a pool of untargeted opponent ships
        pool = list(set(state.opp_ships) - set(self.targets))
        pool = [target for target in pool if self.eligible(target, state)]

        # add new targets with highest values until we have n_targets targets
        while len(self.targets) < n_targets and len(pool) > 0:
            new_target = max(pool, key=lambda x: self.attack_value(x, state))
            pool.remove(new_target)
            if self.attack_value(new_target, state) > 0:
                self.total_bounties += 1
                self.targets.append(new_target)

        # update the target positions, cargos, and rewards
        self.targets_pos = np.array([state.opp_ships[ship][0]
                                     for ship in self.targets]).astype(int)
        self.targets_hal = np.array([state.opp_ships[ship][1]
                                     for ship in self.targets]).astype(int)
        self.rewards = 1000 * np.ones_like(self.targets_pos)
        return

    def eligible(self, target, state):
        # check if the ship is still alive
        if target not in state.opp_ships.keys():
            self.conversions += 1
            self.total_loot += stats.last_state.opp_ships[target][1]
            return False

        # if it got away by getting too close to its own yards to pursue
        if self.dist_to_yard(target, state) <= 1:
            return False

        # if it got away by outrunning its pursuers
        if self.attack_value(target, state) <= 0:  # RETHINK?
            return False

        return True

    def attack_value(self, target, state):
        pos, hal = state.opp_ships[target]

        hunter_pos = np.array([state.my_ships[ship][0] for ship in
                               self.hunters if state.my_ships[ship][1] < hal])
        if hunter_pos.size != 0:
            close_hunters = np.sum(state.dist[hunter_pos, pos] <= 3)
        else:
            close_hunters = 0

        if close_hunters < 3 or self.dist_to_yard(target, state) < 3:
            return -1
        else:
            return close_hunters * hal

    def dist_to_yard(self, target, state):
        # find the yards corresponding to the target ship
        # (state.opp_data is a list of lists...)
        pos = state.opp_ships[target][0]
        opp = [data for data in state.opp_data if pos in data[2]]
        yards = opp[0][1]

        # return the distance of the yards to the position of the ship
        if yards.size == 0:
            return np.inf
        else:
            return np.min(state.dist[yards, pos])


# ------------ CODE IMPORTED FROM fifos.py ------------ #
class Fifos:
    def __init__(self):
        return



# ------------ CODE IMPORTED FROM decide.py ------------ #
def decide(state, actor):
    # legal moves not resulting in self-collision
    no_self_col = [action for action in state.legal_actions(actor)
                   if not state.self_collision(actor, action)]

    # if its a ship
    if actor in state.my_ships:
        # check whether we should convert
        if "CONVERT" in no_self_col and should_convert(actor, state):
            return "CONVERT"

        ship_pos, ship_hal = state.my_ships[actor]

        # construct a weighted graph to calculate distances on
        weights = make_weights(actor, state)
        graph = make_graph_csr(state, weights)

        # calculate the distances from all sites to nearest yard
        yard_dists = dijkstra(graph, indices=state.my_yard_pos, min_only=True)

        # calculate the distance from all sites to the ship
        # also calculate the distance from all sites to the immediate
        # neighbors of ship - then we can easily take a step "towards" any
        # given site later
        nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]
        ship_hood = np.array([state.newpos(ship_pos, move) for move in nnsew])
        hood_dists = dijkstra(graph, indices=ship_hood)
        ship_dists = hood_dists[nnsew.index(None), :]

        # make a map of the relative rewards at each site
        rewards = make_rewards(actor, state)

        # ignore log(0) = -inf warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # the idea is that the expected value of going to any site is the
            # reward at that site, discounted by the time it takes to get to
            # the site and back to the nearest yard
            rate = interest_rate(state, actor)
            values = np.log(ship_hal + rewards) \
                - np.log((1 + rate)) * (1 + yard_dists + ship_dists)

            # don't want to apply this formula for the actual yard locations
            # because it causes ships to idle one step away from the yard
            values[state.my_yard_pos] = np.log(ship_hal) \
                - np.log((1 + rate)) * ship_dists[state.my_yard_pos]

        # the target is the site with the highest reward
        target = values.argmax()

        # prodcuce a list of candidate moves
        candidates = filter_moves(actor, state, target)

        # try to decrease distance to target as much as possible
        dist_after = lambda x: hood_dists[nnsew.index(x), target]
        return min(candidates, key=dist_after)

    # if its a yard
    elif actor in state.my_yards:
        # check if spawning is possible and if we want to
        if "SPAWN" in no_self_col and should_spawn(state, actor):
            return "SPAWN"
        else:
            return None

    # None is always legal - this should never run unless there is a bug
    print(f"BUG at step {1 + state.step}")
    return None


def should_spawn(state, actor=None):
    num_ships = lambda x: x[2].size  # number of ships
    score = lambda x: x[0] + (2/3) * np.sum(x[3])  # halite + 2/3 * cargo
    max_opp_ships = max([num_ships(opp) for opp in state.opp_data])
    max_opp_score = max([score(opp) for opp in state.opp_data])

    my_ships = state.my_ship_pos.size
    my_score = state.my_halite + (2/3) * np.sum(state.my_ship_hal)

    # don't spawn too many more ships towards the end of the game
    # they won't pay for their cost in time
    if state.total_steps - state.step < STEPS_FINAL and my_ships >= 5:
        return False

    # otherwise always keep between 12 and 50 ships
    if my_ships < 12:
        return True
    if my_ships >= 50:
        return False

    # check how many ships are right next to the yard. if there are 3 or more
    # we should wait to spawn until traffic eases
    if actor is not None:
        pos = state.my_yards[actor]
        around = np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1)
        if around >= 3:
            return False

    # after the initial spawning part of the game, always keep enough money
    # to convert a shipyard if necessary
    if state.my_halite < state.config.spawnCost + state.config.convertCost \
            and state.step > STEPS_INITIAL:
        return False

    # keep at least as many ships as the oppponents - cannot get overrun
    # but don't take this too seriously at the end of the game
    offset = 5 * (state.step / state.total_steps)
    if len(state.my_ships) < max_opp_ships - offset:
        return True

    # if we're leading by a lot, keep spawning and increasing dominance
    if my_score > 2000 * (state.step / state.total_steps) + max_opp_score:
        return True

    # in all other cases - don't spawn
    return False


def should_convert(actor, state):
    # check for nr of yard vs nr of ships:
    # do in cluster checks here
    # don't build too many yards if the clustering messes up
    if state.my_yard_pos.size >= \
            round(state.my_ship_pos.size * MAX_YARDS_PER_SHIP):
        return False

    # don't convert too many more ships towards the end of the game
    # they won't pay for their cost in time
    if state.total_steps - state.step < STEPS_FINAL \
            and state.my_yard_pos.size >= 2:
        return False

    # otherwise get the cluster of our ship
    ship_pos, ship_hal = state.my_ships[actor]
    cluster = state.get_cluster(ship_pos)

    # don't build a yard if the ship is an outlier
    if cluster.size == 1:
        return False

    # do build a yard if there is not yard in our cluster and we are the ship
    # in our yard with maximal distance to other yards
    if np.intersect1d(cluster, state.my_yard_pos).size != 0:
        return False

    # build the yard at the ship with maximum distance to other yards
    # if execution gets here cluster_dists will be set
    # yard_inds = np.flatnonzero(np.in1d(state.cluster_pos, state.my_yard_pos))
    # yard_dists = np.min(state.cluster_dists[yard_inds, :], axis=0)

    # if yard_dists[ship_pos] < np.max(yard_dists[cluster]):
    #     return False

    # build yard "evenly spaced" by maximizing l1 distance to others
    yard_dists = np.mean(state.dist[state.my_yard_pos, :], axis=0)
    if yard_dists[ship_pos] < np.max(yard_dists[cluster]):
        return False

    # otherwise we try to build the yard at a site with the most halite
    # sources around - so there will be regeneration in places we control
    # poshal = np.flatnonzero(state.halite_map > 0)
    # dists = state.dist[np.ix_(cluster, poshal)]
    # cauchy = 1 / (1 + dists ** 2)
    # scores = np.sum(cauchy, axis=1)
    # score = scores[cluster == ship_pos]

    # dist = state.dist[ship_pos, poshal]
    # cauchy = 1 / (1 + dist ** 2)
    # score = np.sum(cauchy)

    # if score < np.max(scores):
    #     return False

    return True


def make_rewards(actor, state):
    # the intrisic value of a site is how much halite we can collect there
    rewards = state.config.collectRate * state.halite_map

    # black out any halite next to opposing team's yards
    # our ships can get poached there
    if state.opp_yard_pos.size != 0:
        yards = state.opp_yard_pos
        hood = np.flatnonzero(np.amin(state.dist[yards, :], axis=0) <= 1)
        rewards[hood] = 0

    # check if our ship is a designated hunter and if any bounties have
    # been set. if so, add those targets with higher cargo than ours
    # to the map of rewards
    if actor in bounties.hunters and bounties.not_empty():
        pos, hal = state.my_ships[actor]
        inds = np.flatnonzero(bounties.targets_hal > hal)
        if inds.size != 0:
            rewards[bounties.targets_pos[inds]] = bounties.rewards[inds]

    return rewards


def filter_moves(ship, state, target):
    pos, hal = state.my_ships[ship]

    legal = state.legal_actions(ship)

    # conversion was already considered
    if "CONVERT" in legal:
        legal.remove("CONVERT")

    # no self collisions
    no_self_col = [move for move in legal if not
                   state.self_collision(ship, move)]

    # moves which surely don't result in colliding with an enemy ship
    # (only considers those enemy ships with less halite)
    no_opp_col = [move for move in legal if not
                  state.opp_collision(ship, move, strict=True)]

    # ideally consider moves that don't collide at all
    candidates = list(set(no_self_col) & set(no_opp_col))
    opp_col_flag = False

    # if this is impossible, don't collide with opponent ships
    if len(candidates) == 0:
        candidates = no_opp_col

    # if this is impossible, don't collide with own ships
    if len(candidates) == 0:
        opp_col_flag = True
        candidates = no_self_col

    # if this is impossible, make a legal move
    if len(candidates) == 0:
        candidates = legal

    # if possible, don't idle on shipyards - need room to spawn and deposit
    if pos in state.my_yard_pos and hal == 0:
        if None in candidates and len(candidates) >= 2:
            candidates.remove(None)

    # if we cannot avoid opponent collisions with certainty, choose a move
    # that doesn't stand still, if possible. Since hunters will send one ship
    # onto the square we're occupying, we can get lucky by going to a square
    # the hunters are no longer covering...
    if opp_col_flag:
        if None in candidates and len(candidates) >= 2:
            candidates.remove(None)

    return candidates


def interest_rate(state, ship):
    apos, ahal = state.my_ships[ship]

    # calculate how much space the opponents control with ships that have
    # less halite than we do
    preds = state.opp_ship_pos[state.opp_ship_hal <= ahal]
    if preds.size == 0:
        opp_space = 0
    else:
        opp_sites = np.flatnonzero(np.amin(state.dist[preds, :], axis=0) <= 2)
        opp_space = opp_sites.size

    # set risk proportional to fraction of space controlled by opponent
    risk = 0.1 * opp_space / (state.map_size ** 2)

    # at the very end of the game skyrocket the rate so ships come home
    if state.total_steps - state.step < STEPS_INTEREST_SPIKE:
        risk = 0.8

    # keep a minimum rate of 1 percent even if map is clear
    return 0.01 + risk


def make_weights(actor, state):
    weights = np.ones(state.map_size ** 2)

    # ships contribute weights in the space they control (which is the ball
    # of radius 1 or 2 around their position). sites controlled by multiple
    # ships should get higher weights

    # heuristic: going "through a site" usually takes two steps. if you want
    # to go "around the site" while staying 1 step away it takes 4 steps so
    # the weight should be > 4/2 = 2
    my_weight = GRAPH_MY_WEIGHT

    # again, going "through 3 sites" usually takes 4 steps. if you want to go
    # "around the 3 site" while staying 2 steps from the middle, it takes 8
    # steps so the weight should be > 8/4 = 2. but we want to be very scared
    # of opponent ships so we set this to 4
    opp_weight = GRAPH_OPP_WEIGHT

    apos, ahal = state.my_ships[actor]

    # ignore immediate neighbors in the friendly weights - this is handled
    # automatically and the weights cause traffic jams
    far_ships = state.my_ship_pos[state.dist[apos, state.my_ship_pos] > 1]
    if far_ships.size != 0:
        weights += my_weight * np.sum(state.dist[far_ships, :] <= 1, axis=0)

    # only consider opponent ships with less halite
    less_hal = state.opp_ship_pos[state.opp_ship_hal <= ahal]
    if less_hal.size != 0:
        weights += opp_weight * np.sum(state.dist[less_hal, :] <= 2, axis=0)

    # also need to go around enemy shipyards
    if state.opp_yard_pos.size != 0:
        weights[state.opp_yard_pos] += opp_weight

    # remove any weights on the yards so these don't get blocked
    weights[state.my_yard_pos] = 0

    return weights


def make_graph_csr(state, weights):
    nsites = state.map_size ** 2

    # weight at any edge (x,y) is (w[x] + w[y])/2
    site_weights = weights[state.sites]
    n_weights = 0.5 * (site_weights + weights[state.north])
    s_weights = 0.5 * (site_weights + weights[state.south])
    e_weights = 0.5 * (site_weights + weights[state.east])
    w_weights = 0.5 * (site_weights + weights[state.west])

    # column indices for row i are in indices[indptr[i]:indptr[i+1]]
    # and their corresponding values are in data[indptr[i]:indptr[i+1]]
    indptr = 4 * state.sites
    indptr = np.append(indptr, 4 * nsites)
    indices = np.vstack((state.north, state.south, state.east, state.west))
    indices = np.ravel(indices, "F")
    data = np.vstack((n_weights, s_weights, e_weights, w_weights))
    data = np.ravel(data, "F")

    return csr_matrix((data, indices, indptr), shape=(nsites, nsites))


# ------------ CODE IMPORTED FROM stats.py ------------ #
class Stats:
    def __init__(self):
        # internal controls
        self.state = None
        self.last_state = None

        # historical values - the value at [0] does not exist (nan)
        # self.bla = np.array([np.nan])
        return

    def update(self, arg_state):
        # store our own copy of the state
        self.state = deepcopy(arg_state)

        # if its the first step of the game, and there is no last_state
        # simply set this to state and return
        if self.last_state is None:
            self.last_state = self.state
            return

        # create a new entry in the history and compute various quantities
        # self.bla[step] = ...

        # save the previous state for comparison purposes
        self.last_state = self.state
        return


# ------------ CODE IMPORTED FROM state.py ------------ #
class State:
    def __init__(self, obs, config):
        self.flag = False  # for debugging
        self.config = config  # remove eventually

        # game configuration
        self.map_size = config.size
        self.total_steps = config.episodeSteps
        self.step = obs.step
        self.halite_map = np.array(obs.halite)

        # my halite, yards, and ships
        my_id = obs.player
        self.my_halite, self.my_yards, self.my_ships = obs.players[my_id]

        # set joint and individual data for oppenents
        self._set_opp_data(obs)

        # list of positions with ships that have already moved this turn
        self.moved_this_turn = np.array([])

        # several functions want a vector of all sites so we only generate
        # this once and keep it
        size = self.map_size
        nsites = size ** 2
        self.sites = np.arange(nsites)

        # lookup tables for the effect of moves
        # north[x] is the position north of x, etc
        self.north = (self.sites - size) % nsites
        self.south = (self.sites + size) % nsites
        self.east = size * (self.sites // size) \
            + ((self.sites % size) + 1) % size
        self.west = size * (self.sites // size) \
            + ((self.sites % size) - 1) % size

        # dist[x,y] stores the l1-distance between x and y on the torus
        cols, rows = self.colrow(self.sites)
        coldist = cols - cols[:, np.newaxis]
        rowdist = rows - rows[:, np.newaxis]
        coldist = np.fmin(np.abs(coldist), self.map_size - np.abs(coldist))
        rowdist = np.fmin(np.abs(rowdist), self.map_size - np.abs(rowdist))
        self.dist = coldist + rowdist

        # sets a number of numpy arrays
        self._set_derived()

        # compute clusters used for shipyard conversion
        self._make_clusters()

        return

    def _set_opp_data(self, obs):
        # figure out my id / opponent id
        my_id = obs.player
        opp_ids = list(range(0, len(obs.players)))
        opp_ids.remove(my_id)

        # joint opponent ships and yards
        self.opp_ships = {}
        self.opp_yards = {}
        for opp in opp_ids:
            self.opp_yards.update(obs.players[opp][1])
            self.opp_ships.update(obs.players[opp][2])

        # list of lists. for each opponent has halite, yard positions, ship
        # positions, ship halite as numpy arrays
        self.opp_data = []
        for opp in opp_ids:
            halite, yards, ships = obs.players[opp]
            yard_pos = np.array(list(yards.values())).astype(int)
            if len(ships) > 0:
                poshal = np.array(list(ships.values()))
                ship_pos, ship_hal = np.split(poshal, 2, axis=1)
                ship_pos = np.ravel(ship_pos).astype(int)
                ship_hal = np.ravel(ship_hal).astype(int)
            else:
                ship_pos = np.array([]).astype(int)
                ship_hal = np.array([]).astype(int)

            self.opp_data.append([halite, yard_pos, ship_pos, ship_hal])

        return

    # several function need all ship positions as numpy arrays
    # these arrays need to be set by init() and also updated by update()
    # do this by calling _set_derived()
    def _set_derived(self):
        self.my_yard_pos = np.array(list(self.my_yards.values())).astype(int)
        self.opp_yard_pos = np.array(list(self.opp_yards.values())).astype(int)

        if len(self.my_ships) != 0:
            poshal = np.array(list(self.my_ships.values()))
            self.my_ship_pos, self.my_ship_hal = np.split(poshal, 2, axis=1)
            self.my_ship_pos = np.ravel(self.my_ship_pos).astype(int)
            self.my_ship_hal = np.ravel(self.my_ship_hal).astype(int)
        else:
            self.my_ship_pos = np.array([]).astype(int)
            self.my_ship_hal = np.array([]).astype(int)

        if len(self.opp_ships) != 0:
            poshal = np.array(list(self.opp_ships.values()))
            self.opp_ship_pos, self.opp_ship_hal = np.split(poshal, 2, axis=1)
            self.opp_ship_pos = np.ravel(self.opp_ship_pos).astype(int)
            self.opp_ship_hal = np.ravel(self.opp_ship_hal).astype(int)
        else:
            self.opp_ship_pos = np.array([]).astype(int)
            self.opp_ship_hal = np.array([]).astype(int)

        return

    def _make_clusters(self):
        # array of all sites we are currently occupying
        self.cluster_pos = np.append(self.my_ship_pos, self.my_yard_pos)
        self.cluster_pos = np.unique(self.cluster_pos)


        # if there are not enough sites to cluster, set them all as outliers
        if self.cluster_pos.size <= MIN_CLUSTER_SIZE:
            self.cluster_labels = - np.ones_like(self.cluster_pos)
            return

        # create a graph for clustering - see explanation in make_weights()
        my_weight, opp_weight = GRAPH_MY_WEIGHT, GRAPH_OPP_WEIGHT

        weights = np.ones(self.map_size ** 2)

        weights += my_weight * \
            np.sum(self.dist[self.my_ship_pos, :] <= 1, axis=0)

        if self.opp_ship_pos.size != 0:
            weights += opp_weight * \
                np.sum(self.dist[self.opp_ship_pos, :] <= 2, axis=0)

        if self.opp_yard_pos.size != 0:
            weights[self.opp_yard_pos] += opp_weight

        # unlike make_weights() we DON'T want to remove any weights on
        # the yards for this part
        # weights[state.my_yard_pos] = 0

        graph = make_graph_csr(self, weights)

        # compute graph distances to all cluster sites
        self.cluster_dists = dijkstra(graph, indices=self.cluster_pos)
        dist_matrix = self.cluster_dists[:, self.cluster_pos]

        # run the OPTICS clustering algorithm
        model = OPTICS(min_samples=MIN_SAMPLES,
                       min_cluster_size=MIN_CLUSTER_SIZE,
                       metric="precomputed")

        # ignore outlier warnings from OPTICS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cluster_labels = model.fit_predict(dist_matrix)

        return

    def get_cluster(self, pos):
        if pos not in self.cluster_pos:
            print("Error: cluster site not found")
            return -1

        ind = np.flatnonzero(self.cluster_pos == pos)
        label = self.cluster_labels[ind]

        if label == -1:  # outliers - return only the position
            return np.array([pos])
        else:
            cluster_inds = np.flatnonzero(self.cluster_labels == label)
            return self.cluster_pos[cluster_inds]

    def actors(self):
        return list(self.my_ships.keys()) + list(self.my_yards.keys())

    def colrow(self, pos):
        return pos % self.map_size, pos // self.map_size

    def newpos(self, pos, action):
        if (action is None) or (action == "CONVERT"):
            return pos
        elif action == "NORTH":
            return self.north[pos]
        elif action == "SOUTH":
            return self.south[pos]
        elif action == "EAST":
            return self.east[pos]
        elif action == "WEST":
            return self.west[pos]

    def update(self, actor, action):
        # first check if actor is yard and act accordingly
        if (actor in self.my_yards) and (action is None):
            return

        if action == "SPAWN":
            # generate a unique id string
            newid = f"spawn[{str(actor)}]-{str(uuid4())}"

            # create a new ship at yard position
            pos = self.my_yards[actor]
            self.my_ships[newid] = [pos, 0]

            # the result is a ship here that cannot move this turn
            self.moved_this_turn = np.append(self.moved_this_turn, pos)

            # subtract spawn cost from available halite
            self.my_halite -= int(self.config.spawnCost)

            # update internal variables that derive from my_ships, my_yards
            self._set_derived()
            return

        # if execution gets here, we know actor is a ship
        pos, hal = self.my_ships[actor]

        if action == "CONVERT":
            # generate a unique id string
            newid = f"convert[{str(actor)}]-{str(uuid4())}"

            # create a new yard at ship position and remove ship
            self.my_yards[newid] = pos
            del self.my_ships[actor]

            # remove conversion cost from available halite but don't add any
            # net gains - its not available until next turn
            self.my_halite += int(max(hal - self.config.convertCost, 0))
            self.halite_map[pos] = 0

            # update internal variables that derive from my_ships, my_yards
            self._set_derived()
            return

        # if execution gets here, the actor stays a ship
        # first update the halite
        # if the ship stays put, it can collect halite
        if action is None:
            collect = np.floor(self.halite_map[pos] * self.config.collectRate)
            nhal = int(hal + collect)
            self.halite_map[pos] -= collect

        # otherwise the move may cost halite
        else:
            nhal = int(hal * (1 - self.config.moveCost))

        # now update the position, and write the new values into my_ships
        npos = self.newpos(pos, action)
        self.my_ships[actor] = [npos, nhal]

        # add position to list of ships that can no longer move this turn
        self.moved_this_turn = np.append(self.moved_this_turn, npos)

        # update internal variables that derive from my_ships, my_yards
        self._set_derived()
        return

    def legal_actions(self, actor):
        # if the actor is a yard
        if actor in self.my_yards:
            if self.my_halite < self.config.spawnCost:
                return [None]
            else:
                return [None, "SPAWN"]

        # if execution gets here, actor is a ship
        actions = [None, "CONVERT", "NORTH", "SOUTH", "EAST", "WEST"]
        pos, hal = self.my_ships[actor]

        # can't convert if you don't have enough halite or are in a yard,
        if self.my_halite < self.config.convertCost - hal \
                or pos in self.my_yards.values():
            actions.remove("CONVERT")

        # if you have no ships and no halite to build new ones, you lose
        elif len(self.my_ships) == 1 \
                and self.my_halite < self.config.spawnCost:
            actions.remove("CONVERT")

        return actions

    def self_collision(self, actor, action):
        # shipyards only give collisions if we spawn a ship after moving a ship
        # there previously
        if actor in self.my_yards:
            pos = self.my_yards[actor]
            if action == "SPAWN" and pos in self.moved_this_turn:
                return True
            else:
                return False

        # if execution gets here, it's a ship
        pos, hal = self.my_ships[actor]

        # cannot convert onto another shiyard - also checked by legal_moves()
        if action == "CONVERT":
            if pos in self.my_yards.values():
                return True
            else:
                return False

        # otherwise get the new position of the ship and check if it runs into
        # a ship that can no longer move
        npos = self.newpos(pos, action)
        if npos in self.moved_this_turn:
            return True

        return False

    def opp_collision(self, ship, action, strict=True):
        # return True if a collision is possible in the next step if ship
        # performs action

        pos, hal = self.my_ships[ship]
        npos = self.newpos(pos, action)

        # if there are no opponent ships, just check whether we run into a yard
        if self.opp_ship_pos.size == 0:
            return (True if npos in self.opp_yard_pos else False)

        # if execution gets here, self.opp_ship_pos is not empty
        # find those ships that have less halite than we do
        if strict is True:
            less_hal = self.opp_ship_pos[self.opp_ship_hal <= hal]
        else:
            less_hal = self.opp_ship_pos[self.opp_ship_hal < hal]

        # less_hal could still be empty - if so, again, just check whether we
        # run into a yard
        if less_hal.size == 0:
            return (True if npos in self.opp_yard_pos else False)

        # if execution gets here, less_hal is not empty
        # list of sites where ships in less_hal could end up after one turn
        hood = np.flatnonzero(np.amin(self.dist[less_hal, :], axis=0) <= 1)

        # regard our own shipyards as safe havens - this is not necessarily
        # the case but many opponents will not want to run into the yard
        # this will cost them their ship too
        hood = np.setdiff1d(hood, self.my_yard_pos)

        if npos in self.opp_yard_pos or npos in hood:
            return True

        return False

    def opp_ship_collision(self, ship, action, strict=True):
        # return True if a collision only with opponent SHIPS (not yards) is
        # possible in the next step if ship performs action. this is for
        # attacking shipyards

        pos, hal = self.my_ships[ship]
        npos = self.newpos(pos, action)

        if strict:
            less_hal = self.opp_ship_pos[self.opp_ship_hal <= hal]
        else:
            less_hal = self.opp_ship_pos[self.opp_ship_hal < hal]

        # less_hal could be empty
        if less_hal.size == 0:
            return False

        # if execution gets here, less_hal is not empty
        # list of sites where ships in less_hal could end up after one turn
        hood = np.flatnonzero(np.amin(self.dist[less_hal, :], axis=0) <= 1)

        # regard our own shipyards as safe havens - this is not necessarily
        # the case but many opponents will not want to run into the yard
        # this will cost them their ship too
        hood = np.setdiff1d(hood, self.my_yard_pos)

        return True if npos in hood else False


# ------------ CODE IMPORTED FROM init.py ------------ #
# global statistics object that needs to preserved over all turns
# gets updated only by agent() using stats.update()
# all other functions should be read-only
stats = Stats()
bounties = Bounties()
fifos = Fifos()


# ------------ CODE IMPORTED FROM agent.py ------------ #
def agent(obs, config):
    # keep track of time to prevent agent timeouts
    tick = time()  # first tock before the main loop while queue.not_empty()

    # internal game state, to be updated as we decide on actions
    state = State(obs, config)

    # update the global statistics we keep track of across all turns
    # global stats
    # stats.update(state)

    # update which opponent ships should be targeted by our hunters
    global bounties
    bounties.update(state)

    # list of ships/yards for which we need to decide on an action
    queue = Queue(state.actors())

    # check if there are any urgent actions to be taken to ensure survival
    # if yes, return these actions and update state and queue
    # if no, state and queue do not change and actions = {} is initialized
    # to an empty dictionary
    actions = survival(state, queue)

    # now decide on "normal" actions for the remaining actors (if any)
    # while there are ships/yards to receive a command
    tock = time()
    while queue.not_empty():
        # agent has to make decisions in a certain amount of time
        if tock > tick + state.config.actTimeout - 0.1:
            print(f"Agent timeout at step {state.step + 1}")
            break

        # schedule the next ship/yard
        actor = queue.schedule(state)

        # decide on an action for it
        action = decide(state, actor)

        # update game state with consequence of action
        state.update(actor, action)

        # write action into dictionary of actions to return
        if action is not None:
            actions[actor] = action

        tock = time()

    # MOVE BACK
    global stats
    stats.update(state)

    if state.step == 398:
        print(f"converted {bounties.conversions} / {bounties.total_bounties}")
        print(f"total loot: {bounties.total_loot}")

    print(f"Deuce: Step {1+state.step} took {tock - tick} seconds")

    return actions
