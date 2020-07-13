#### CODE IMPORTED FROM leading.py ####
"""Initialization code.

Put any imports you need in the agent code here, along with any code that
should run immediately (and only once) at the beginning of the episode,
For now we just say hi - but I could imagine saving the state in some global
variables declared here, etc.
"""


import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sklearn.cluster import OPTICS

from uuid import uuid4
from time import time


# agent startup code needs to run in < 10 seconds
# print("Ahoy!")


#### CODE IMPORTED FROM queue.py ####
"""The scheduling logic of the agent.

Defines:
    Queue - rudimentary class with actors pending a decision
"""


class Queue:
    """Stores a list of actors pending a decision.

    Queue is initialized with a list of keystrings of ships/yards.

    The schedule() method returns the ship/yard with the highest priority
    and removes it from the list of pending ships/yards.
    """

    def __init__(self, initkeys=[]):
        self.pending = initkeys
        return

    def not_empty(self):
        return len(self.pending) > 0

    def remove(self, actor):
        if actor in self.pending:
            self.pending.remove(actor)
        return

    def schedule(self, state):
        """Decide which of the remaining pending ships/yards has the highest
        priority and returns its keystring
        """

        # separate actors into ships and yards
        ships = [actor for actor in self.pending if actor in state.my_ships]
        yards = [actor for actor in self.pending if actor in state.my_yards]

        # if there are still ships, schedule the ship with the most halite
        # otherwise, take the first yard
        if len(ships) > 0:
            halite = lambda x: state.my_ships[x][1]
            nextup = max(ships, key=halite)
        # else we have only yards left. take yards with small clusters first
        # so ships will spawn where there is the most space
        else:
            clust_size = lambda x: state.get_cluster(state.my_yards[x]).size
            nextup = min(yards, key=clust_size)

        # remove the scheduled actor from the pending list and return it
        self.remove(nextup)
        return nextup


#### CODE IMPORTED FROM survival.py ####
"""Ensuring survival etc"""


# right now it just builds a ship or a yard if there are none
# and we have money.
def survival(state, queue):
    actions = {}

    if len(state.my_yards) == 0:
        print(f"No shipyards at step {state.step + 1}")

        # see if it is possible to convert the ship with the most cargo
        # if so, get it by schedule(), convert it and update state
        max_cargo = state.my_ship_hal.max()
        if max_cargo + state.my_halite >= state.config.convertCost:
            ship = queue.schedule(state)
            actions[ship] = "CONVERT"
            state.update(ship, "CONVERT")
        # otherwise instruct all ships to survive as long as possible while
        # collecting as much halite as possible
        else:
            actions = run_and_collect(state, queue)

    # do elif since it can never happen that we start the turn without any
    # ships or yards - but it can happen that the previous block converted
    # our last ship...
    elif len(state.my_ships) == 0:
        print(f"No ships at step {state.step + 1}")

        # spawn as many ships as we can afford
        actions = spawn_maximum(state, queue)

    if len(actions) > 0:
        print("The following actions were taken")
        print(actions)

    return actions


def run_and_collect(state, queue):
    actions = {}

    while queue.not_empty():
        # schedule the next ship

        ship = queue.schedule(state)
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


#### CODE IMPORTED FROM decide.py ####
"""Contains the decision logic of the agent.

Defines the function decide(), which takes a game state and an actor. The
output is the action that should be taken by actor.
"""


def decide(state, actor):

    """Decide which action should be taken by the ship/yard actor."""
    # set rates depending on game situation
    rates = set_rates(actor, state)

    # initially, the candidates are a list of legal actions that don't result
    # in self-collision but we revert to legal actions below if non-collision
    # is impossible
    legal = state.legal_actions(actor)
    no_self_col = [action for action in legal
                  if not state.self_collision(actor, action)]

    # if its a ship
    if actor in state.my_ships:
        ship_pos, ship_hal = state.my_ships[actor]

        # lookup tables to translate between moves and positions
        nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]
        ship_hood = [state.newpos(ship_pos, move) for move in nnsew]

        # construct a weighted graph to calculate distances on
        weights = make_weights(actor, state, rates)
        graph = make_graph_csr(state, weights)

        # calculate the distances from all sites to nearest yard
        yard_dists, preds, srcs = dijkstra(graph,
                                           indices=state.my_yard_pos,
                                           min_only=True,
                                           return_predecessors=True)

        # get the unweighted length of the shortest path from ship_pos to the
        # nearest yard by following the predecessor list back to the source
        # this may not be necessary since the interest spike does a good job...
        time_to_yard = 0
        ind = ship_pos
        while preds[ind] not in [srcs[ship_pos], -9999]:
            time_to_yard += 1
            ind = preds[ind]

        # check whether we should convert
        if "CONVERT" in no_self_col:  # same as "CONVERT" being legal
            if should_convert(actor, state, rates, time_to_yard, yard_dists):
                print(f"Conversion at step {1 + state.step}")
                return "CONVERT"
            else:
                legal.remove("CONVERT")  # see next conditional
                no_self_col.remove("CONVERT")

        # decide on a list of allowable moves. We would like, in this order:
        # no self collisions and no opp collisions, no opp collisions, no self
        # collisions,
        no_opp_col = [action for action in legal
                      if not state.opp_collision(actor, action)]

        candidates = list(set(no_self_col) & set(no_opp_col))

        if len(candidates) == 0:
            candidates = no_opp_col

        if len(candidates) == 0:
            print(f"Survival not guaranteed: {1 + state.step}")
            candidates = no_self_col

        if len(candidates) == 0:
            candidates = legal

        # if time to go home is larger than time remaining (+ some buffer)
        # then we want to go home on the shortest path that avoids collision
        time_remaining = state.config.episodeSteps - state.step
        if time_to_yard > 4 + time_remaining:
            dist_after = lambda x: yard_dists[ship_hood[nnsew.index(x)]]
            return min(candidates, key=dist_after)

        # if none of the preceeding special cases happen, we calculate
        # a value for each site and go towards the site with the highest value
        # make a map of the relative rewards at each site
        effmap = make_map(actor, state, weights, rates)

        # calculate the distance from all sites to the ship
        # also calculate the distance from all sites to the immediate
        # neighbors of ship - then we can easily take a step "towards" any
        # given site later
        hood_dists = dijkstra(graph, indices=ship_hood)
        ship_dists = hood_dists[nnsew.index(None), :]

        # the idea is that the expected value of going to any site is the
        # reward at that site, discounted by the time it takes to get to the
        # site and back to the nearest yard
        rate = rates["risk"]
        values = np.log(0.01 + ship_hal + effmap) \
            - np.log((1 + rate)) * (1 + yard_dists + ship_dists)

        # don't want to apply this formula for the actual yard locations
        # because it causes ships to idle one step away from the yard
        values[state.my_yard_pos] = np.log(0.01 + ship_hal) \
            - np.log((1 + rate)) * ship_dists[state.my_yard_pos]

        # the target is the site with the highest reward
        target = values.argmax()

        # try to decrease distance to target as much as possible
        dist_after = lambda x: hood_dists[nnsew.index(x), target]
        return min(candidates, key=dist_after)

    # if its a yard
    elif actor in state.my_yards:
        # check if spawning is possible and if we want to
        if "SPAWN" in no_self_col and should_spawn(actor, state, rates):
            return "SPAWN"
        else:
            return None

    # None is always legal - this should never run unless there is a bug
    print(f"BUG at step {1 + state.step}")
    return None


# needs more work
def should_convert(actor, state, rates, time_to_yard, yard_dists):
    ship_pos, ship_hal = state.my_ships[actor]

    # see if there is no chance of coming home. if we have more than
    # convertCost in cargo, should still convert and keep difference
    time_remaining = state.config.episodeSteps - state.step
    if time_to_yard > 0.3 * time_remaining and \
            ship_hal > state.config.convertCost:
        return True

    # otherwise get the cluster of our ship
    cluster = state.get_cluster(ship_pos)

    # don't build a yard if the ship is an outlier
    if cluster.size == 1:
        return False

    # do build a yard if there is not yard in our cluster and we are the ship
    # in our yard with maximal distance to other yards

    if np.intersect1d(cluster, state.my_yard_pos).size != 0:
        # c, r = state.colrow(ship_pos)
        # print(f"Me: ({c}, {r})")
        # print("My cluster:")
        # c, r = state.colrow(cluster)
        # print(np.hstack([c[:, np.newaxis], r[:, np.newaxis]]))
        # print("Converting!")
        return False

    # build the yard at the ship with maximum distance to other yards
    if yard_dists[ship_pos] < np.max(yard_dists[cluster]):
        return False

    return True


def should_spawn(actor, state, rates):
    if ships_wanted(state, rates) < len(state.my_ships):
        return False

    # don't spawn any more ships towards the end of the game
    # they won't pay for their cost in time
    if state.config.episodeSteps - state.step < state.map_size:
        return False

    # check how many ships are at distance one or two from the yard
    # and return False if there are too many
    pos = state.my_yards[actor]
    dist1 = np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1)
    # dist2 = np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 2)
    if dist1 > 2: # or dist2 > 2:
        return False

    # otherwise, we want a ship
    return True


def ships_wanted(state, rates):
    # as long as there is free space, we want to send ships to occupy it
    # each ship controls 13 sites since |B(0,2)_L1| = 13 and we want to
    # evenly cover space we control and free space

    # if rates["my_space"] < rates["opp_space"]/(0.1 + state.opps_left):
    #     S = (rates["free_space"] + rates["my_space"]) / 13
    # else:
    #     S = rates["free_hal"] \
    #         / ((1 + state.opps_left) * state.config.spawnCost) \
    #         + rates["my_space"] / 13

    # if rates["my_space"] > rates["opp_space"] / (0.1 + state.opps_left):
    #     S = rates["free_hal"] \
    #         / ((0.1 + state.opps_left) * state.config.spawnCost) \
    #         + rates["my_space"] / 13
    # else:
    #     S = (rates["free_space"] + rates["my_space"]) / 16

    # S = rates["free_hal"] \
    #         / ((0.1 + state.opps_left) * state.config.spawnCost) \
    #         + rates["my_space"] / 13



    if state.my_halite > 2000 * (state.step/state.config.episodeSteps) \
                       + max(state.opp_scores):
        S = (rates["free_space"] + rates["my_space"]) / 10
    else:
        S = (rates["free_space"] + rates["my_space"]) / 16

    S = max(10, S)
    return S


def set_rates(actor, state):
    rates = {}

    # calculate how much space we control versus opponents
    # any ship or yard controls all sites in B(x, 2)_L1 where
    # x is its position
    opp_sites = np.append(state.opp_ship_pos, state.opp_yard_pos)
    if opp_sites.size != 0:
        opp_sites = np.where(np.amin(state.dist[opp_sites, :], axis=0) <= 2)
        opp_sites = np.unique(opp_sites)

    my_sites = np.append(state.my_ship_pos, state.my_yard_pos)
    if my_sites.size != 0:  # should not be necessary, kaggle bug?
        my_sites = np.where(np.amin(state.dist[my_sites, :], axis=0) <= 2)
        my_sites = np.unique(my_sites)

    free = np.setdiff1d(state.sites, np.union1d(my_sites, opp_sites))
    overlap = np.intersect1d(opp_sites, my_sites)
    opp_sites = np.setdiff1d(opp_sites, overlap)
    my_space = np.setdiff1d(my_sites, overlap)

    rates["my_space"] = my_sites.size
    rates["opp_space"] = opp_sites.size
    rates["contested"] = overlap.size
    rates["free_space"] = free.size

    # how much halite is available on the entire map and in the free space
    rates["total_hal"] = np.sum(state.halite_map)
    rates["free_hal"] = 0 if free.size == 0 \
        else np.sum(state.halite_map[free])
    rates["my_hal"] = np.sum(state.halite_map[my_space])

    # risk due to other ships eating our guys proportional to space controlled
    # by opponent. this may be a little crude - think more
    risk = 0.1 * rates["opp_space"] / (state.map_size ** 2)

    # at the very end of the game skyrocket the rate so ships come home
    if state.config.episodeSteps - state.step < 10:
        risk = 0.8

    # keep a minimum rate of 5% even if we're dominating to keep depositing
    rates["risk"] = 0.01 + risk
    # print(rates["risk"])

    return rates


def make_map(actor, state, weights, rates):
    # the intrisic value of a site is how much halite we can collect there
    effmap = state.config.collectRate * state.halite_map

    # we can attack all opponent ships that have more halite than we do
    # so we add their cargos to the value of their positions.
    # but we expect the ship to move in the time it takes for us to get there
    # so we spread this out evenly in the smallest ball around the ship
    # that also contains our position
    apos, ahal = state.my_ships[actor]

    attack_quantile = max(0.01, 1 - np.sum(state.halite_map) / state.config.startingHalite)

    if ahal < np.quantile(state.my_ship_hal, attack_quantile):
        more_hal = np.flatnonzero(state.opp_ship_hal > ahal)
        dists = state.dist[apos, state.opp_ship_pos[more_hal]]
        effmap[state.opp_ship_pos[more_hal]] = state.opp_ship_hal[more_hal] \
            / (1 + dists ** 2)


    # for ind in more_hal:
    #     pos = state.opp_ship_pos[ind]
    #     rad = state.dist[pos, apos]
    #     vol = 1 + 2 * rad * (rad + 1)  # = volume of 2d L1 ball
    #     dists = state.dist[pos, :]
    #     effmap[dists <= rad] += state.opp_ship_hal[ind] / vol

    # NOT sure about this
    # we can also destroy an opponents shipyard if we run into it this costs
    # us spawnCost + cargo and them spawnCost + convertCost. Its worth it for
    # us if cargo < convertCost / # of opps, since all opponents benefit...
    # but we only do this if we have a lot of ships to spare - this follows
    # the same logic as in should_spawn()
    # if ahal < state.config.convertCost / (1 + state.opps_left) \
    #         and ships_wanted(state, rates) <= len(state.my_ships):
    #     if state.opp_yard_pos.size != 0:
    #         effmap[state.opp_yard_pos] = \
    #             (state.config.convertCost / (1 + state.opps_left)) - ahal

    return effmap


def make_weights(actor, state, rates):
    weights = np.ones(state.map_size ** 2)

    # heuristic: going "through a site" usually takes two steps. if you want
    # to go "around the site" while staying 1 step away it takes 4 steps so
    # the weight should be > 4/2 = 2. but we make it a little higher if there
    # is a lot of halite in the free space to encourage our ships to spread
    my_weight = (2 + rates["free_hal"] / rates["total_hal"])

    # again, going "through 3 sites" usually takes 4 steps. if you want to go
    # "around the 3 site" while staying 2 steps from the middle, it takes 8
    # steps so the weight should be > 8/4 = 2. but we want to be very scared
    # of opponent ships so we set this to 4
    opp_weight = 4

    apos, ahal = state.my_ships[actor]

    # ships contribute weights in the space they control (which is the ball
    # of radius 2 around their position). But we want to count ships twice.
    # DO this without looping!!!
    my_other_ships = np.setdiff1d(state.my_ship_pos, apos)
    for pos in my_other_ships:
        # don't set weights for very close ships causing traffic jams
        # setting this bigger reduces spreading out though...
        if state.dist[pos, apos] <= 2:
            continue
        dists = state.dist[pos, :]
        weights[dists <= 1] += my_weight

    # only consider opponent ships with less halite
    less_hal = state.opp_ship_pos[state.opp_ship_hal <= ahal]
    for pos in less_hal:
        dists = state.dist[pos, :]
        weights[dists <= 2] += opp_weight

    # weights[state.opp_yard_pos] += opp_weight
    # NOT sure about this
    # same logic as in make_map - this is the opposite condition
    # if ahal > state.config.convertCost / (1 + state.opps_left) \
    #         or ships_wanted(state, rates) > len(state.my_ships):
    #     if state.opp_yard_pos.size != 0:
    #         weights[state.opp_yard_pos] += opp_weight

    return weights


def make_graph_csr(state, weights):
    """Returns the graph in csr format. Vertices are adjacent if they are
    adjacent on the torus graph. The weight on edge [x,y] is (w[x] + w[y])/2.
    """
    nsites = state.map_size ** 2

    # weight at any edge (x,y) is (w[x] + w[y])/2
    site_weights = weights[state.sites]
    north_weights = 0.5 * (site_weights + weights[state.north])
    south_weights = 0.5 * (site_weights + weights[state.south])
    east_weights = 0.5 * (site_weights + weights[state.east])
    west_weights = 0.5 * (site_weights + weights[state.west])

    # column indices for row i are in indices[indptr[i]:indptr[i+1]]
    # and their corresponding values are in data[indptr[i]:indptr[i+1]]
    indptr = 4 * state.sites
    indptr = np.append(indptr, 4 * nsites)
    indices = np.vstack((state.north, state.south, state.east, state.west))
    indices = np.ravel(indices, "F")
    data = np.vstack((north_weights, south_weights, east_weights, west_weights))
    data = np.ravel(data, "F")

    return csr_matrix((data, indices, indptr), shape=(nsites, nsites))


#### CODE IMPORTED FROM state.py ####
"""Contains the mechanical parts of the agents decision making.

Defines the GameState class, which encodes the internal state of the game. It
has an update() method, that updates the internal state with the effect of
actor performing action. It also defines legal_moves() which returns a list
of legal moves for the actor.
"""


class State:
    """The internal state of the game.

    GameState is initialized with the (obs, config) pair passed to agent()
    by the kaggle environment. The class implements various methods for
    processing the information stored in this pair. In addition, it provides
    the method update(shiporyard,action), which updates the (obs, config) pair
    with the effect of performing action with shiporyard. Finally, it provides
    the method collision(shiporyard, action) which returns whether the action
    results in a collision together with some information on the outcome of
    such a collision.
    """

    def __init__(self, obs, config):
        self.flag = False  # for debugging
        self.config = config  # remove eventually

        # game configuration
        self.map_size = config.size
        self.total_steps = config.episodeSteps
        self.halite_map = np.fmax(0, np.array(obs.halite))
        self.moved_this_turn = np.array([])
        self.step = obs.step

        # my halite, yards, and ships
        my_id = obs.player
        self.my_halite, self.my_yards, self.my_ships = obs["players"][my_id]

        # opponent yards and ships
        opp_ids = list(range(0, len(obs.players)))
        opp_ids.remove(my_id)
        self.opp_ships = {}
        self.opp_yards = {}
        self.opp_scores = []
        for opp in opp_ids:
            self.opp_yards.update(obs.players[opp][1])
            self.opp_ships.update(obs.players[opp][2])
            self.opp_scores.append(obs.players[opp][0])

        self.opps_left = len([None for opp in opp_ids
                              if len(obs.players[opp][1]) != 0
                              or len(obs.players[opp][2]) != 0])

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

        self._set_derived()

        self._make_clusters()

        return

    def _make_clusters(self):
        # array of all sites we are currently occupying
        self.cluster_pos = np.append(self.my_ship_pos, self.my_yard_pos)
        self.cluster_pos = np.unique(self.cluster_pos)

        # choose this more intelligently?
        MIN_SIZE = 5

        # if there are not enough sites to cluster, set them all as outliers
        if self.cluster_pos.size <= MIN_SIZE:
            self.cluster_labels = - np.ones_like(self.cluster_pos)
            return

        # create a graph for clustering
        # see logic in make_weights()
        weights = np.ones(self.map_size ** 2)

        for pos in self.my_ship_pos:
            dists = self.dist[pos, :]
            weights[dists <= 1] += 2

        for pos in self.opp_ship_pos:
            dists = self.dist[pos, :]
            weights[dists <= 2] += 4

        weights[self.opp_yard_pos] += 4

        graph = make_graph_csr(self, weights)

        site_dists = dijkstra(graph, indices=self.cluster_pos,
                              return_predecessors=False, directed=False)
        site_dists = site_dists[:, self.cluster_pos]

        # launch 4 and 5 here, see some stuff
        model = OPTICS(min_samples=3, min_cluster_size=5, max_eps=0.3*site_dists.max(), metric="precomputed")

        self.cluster_labels = model.fit_predict(site_dists)

        return


    def get_cluster(self, pos):
        if pos not in self.cluster_pos:
            print("Error: cluster site not found")
            return -1

        ind = np.where(self.cluster_pos == pos)
        label = self.cluster_labels[ind]

        if label == -1:
            return np.array([pos])
        else:
            cluster_inds = np.where(self.cluster_labels == label)
            return self.cluster_pos[cluster_inds]


    # several function need all ship positions as numpy arrays
    # these arrays need to be set by init() and also updated by update()
    # do this by calling _set_derived()
    def _set_derived(self):
        self.my_yard_pos = np.array(list(self.my_yards.values())).astype(int)
        self.opp_yard_pos = np.array(list(self.opp_yards.values())).astype(int)

        if len(self.my_ships) > 0:
            poshal = np.array(list(self.my_ships.values()))
            self.my_ship_pos, self.my_ship_hal = np.split(poshal, 2, axis=1)
            self.my_ship_pos = np.ravel(self.my_ship_pos).astype(int)
            self.my_ship_hal = np.ravel(self.my_ship_hal).astype(int)
        else:
            self.my_ship_pos = np.array([]).astype(int)
            self.my_ship_hal = np.array([]).astype(int)

        if len(self.opp_ships) > 0:
            poshal = np.array(list(self.opp_ships.values()))
            self.opp_ship_pos, self.opp_ship_hal = np.split(poshal, 2, axis=1)
            self.opp_ship_pos = np.ravel(self.opp_ship_pos).astype(int)
            self.opp_ship_hal = np.ravel(self.opp_ship_hal).astype(int)
        else:
            self.opp_ship_pos = np.array([]).astype(int)
            self.opp_ship_hal = np.array([]).astype(int)

        return

    def actors(self):
        """Return all ship/yard keys."""
        return list(self.my_ships.keys()) + list(self.my_yards.keys())

    def colrow(self, pos):
        """Linear position pos as (column, row) pair."""
        return pos % self.map_size, pos // self.map_size

    def newpos(self, pos, action):
        """Return new position when ship at pos performs action."""
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
        """Update the state when actor performs action."""
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
        if action is None:  # can collect or deposit halite
            if pos in self.my_yards.values():
                # Don't update my_halite. Subsequent ships/yards with think
                # its available for SPAWN/CONVERT but its not
                # self.my_halite += hal
                nhal = 0
            else:  # collect halite
                collect = np.floor(self.halite_map[pos]
                                   * self.config.collectRate)
                nhal = int(hal + collect)
                self.halite_map[pos] -= collect

        else:  # the action is a real move that may cost halite
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
        """Generate legal actions for actor."""
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
        """Return True if (actor, action) results in collision among our ships.

        Only return True if there will be a collision at the end of the turn.
        So if (actor, action) puts actor on a cell occupied by a ship that has
        not yet moved, the return value is False. However, this will not catch
        moves that result in collision because other ships have nowhere left to
        go... This extension may be doable...
        """
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


    def opp_collision(self, ship, action):
        # return True if a collision is possible in the next step if ship
        # performs action

        pos, hal = self.my_ships[ship]
        npos = self.newpos(pos, action)

        # if there are no opponent ships, just check whether we run into a yard
        if self.opp_ship_pos.size == 0:
            return (True if npos in self.opp_yard_pos else False)

        # if execution gets here, self.opp_ship_pos is not empty
        # find those ships that have less halite than we do
        less_hal = self.opp_ship_pos[self.opp_ship_hal <= hal]

        # less_hal could still be empty - if so, again, just check whether we
        # run into a yard
        if less_hal.size == 0:
            return (True if npos in self.opp_yard_pos else False)

        # if execution gets here, less_hal is not empty
        # list of sites where ships in less_hal could end up after one turn
        ship_hood = np.where(np.amin(self.dist[less_hal, :], axis=0) <= 1)
        ship_hood = np.unique(ship_hood)

        if npos in self.opp_yard_pos or npos in ship_hood:
            return True

        return False


#### CODE IMPORTED FROM agent.py ####
"""
The function called by the kaggle environment at each turn.

It converts the (obs, config) pair into a GameState class and defines an
ActionQueue with all of our ships/yards. It then repeats a loop that schedules
the highest priority ship, decides on an action for this ship, updates the
game state, and writes the action into a list to be returned to the kaggle
environment.
"""


def agent(obs, config):
    """Agent function called by the halite environment.

    This contains the top-level logic -- it gets a game state in (obs, config)
    and returns a list of actions.
    """
    # keep track of time to prevent agent timeouts
    tick = time()  # first tock before the main loop while queue.not_empty()

    # internal game state, to be updated as we decide on actions
    state = State(obs, config)

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
        # agent has to make decisions in < 6 seconds
        if tock > tick + state.config.actTimeout - 0.1:
            print(f"Timeout at step {state.step + 1}")
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

    # print(f"Step {state.step + 1} took {round(tock-tick, 3)} seconds\n\n")
    return actions
