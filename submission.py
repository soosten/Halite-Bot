# ------------ CODE IMPORTED FROM imports.py ------------ #
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import OPTICS
from uuid import uuid4
import warnings


# ------------ CODE IMPORTED FROM queue.py ------------ #
class Queue:
    def __init__(self, state):
        self.ships = list(state.my_ships.keys())
        self.yards = list(state.my_yards.keys())
        return

    def pending(self):
        return len(self.ships) + len(self.yards) > 0

    def remove(self, actor):
        if actor in self.ships:
            self.ships.remove(actor)
        if actor in self.yards:
            self.yards.remove(actor)
        return

    def schedule(self, state):
        # if there are ships, schedule the ship with the highest priority
        if len(self.ships) > 0:
            nextup = max(self.ships, key=lambda s: self.priority(state, s))

        # else we have only yards left. take yards with small clusters first
        # so ships will spawn where there is the most space
        else:
            clust_size = lambda y: state.get_cluster(state.my_yards[y]).size
            nextup = min(self.yards, key=clust_size)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup

    def priority(self, state, ship):
        # ships have highest priority if they are likely to convert
        if should_convert(ship, state):
            return 2000

        # next-highest priority if they have one or less non-colliding moves
        moves = [None, "NORTH", "SOUTH", "EAST", "WEST"]
        collision = lambda m: state.opp_collision(ship, m) \
            or state.self_collision(ship, m)
        if len([1 for move in moves if not collision(move)]) <= 1:
            return 1000

        # otherwise they are ranked by cargo
        return state.my_ships[ship][1]


# ------------ CODE IMPORTED FROM targets.py ------------ #
class Targets:
    def __init__(self):
        self.hunters = {}
        self.hunters_pos = np.array([]).astype(int)
        self.hunters_hal = np.array([]).astype(int)

        self.ship_targets = []
        self.ship_targets_pos = np.array([]).astype(int)
        self.ship_targets_hal = np.array([]).astype(int)
        self.ship_targets_rew = np.array([]).astype(int)

        # remove
        self.total_bounties = 0
        self.conversions = 0
        self.total_loot = 0

        return

    def update(self, state):
        self.set_hunters(state)
        self.set_ship_targets(state)
        self.set_yard_targets(state)
        return

    def set_hunters(self, state):
        # determine hunters depending on cargo and halite on the map
        if state.my_ship_pos.size == 0:
            self.hunters = {}
        else:
            # determine a mining value of each ship based on its cargo
            # and the amount of halite near it
            # for each ship position x, hood_hal[x] is the average halite in
            # the cells around x. we add this to the cargo to break ties in
            # values of ships with equal cargo
            boolhood = state.dist[state.my_ship_pos, :] <= 2
            hood = np.apply_along_axis(np.flatnonzero, 1, boolhood)
            hood_hal = np.mean(state.halite_map[hood], axis=1)
            ship_val = state.my_ship_hal + 0.1 * hood_hal

            # choose a fraction of ships depending on halite on the map
            q = MAX_HUNTERS_PER_SHIP
            q -= np.sum(state.halite_map) / state.config.startingHalite
            q = q if q > 0.3 else 0
            num_hunters = int(q * len(state.my_ships))

            # designate the num_hunters ships with lowest value as hunters
            hunters_ind = np.argpartition(ship_val, num_hunters)
            hunters_ind = hunters_ind[0:num_hunters]
            self.hunters_pos = state.my_ship_pos[hunters_ind]

            # remove any fifo ships from the hunters
            self.hunters_pos = np.setdiff1d(self.hunters_pos, fifos.fifo_pos)

            # also keep hunters in dictionary form like state.my_ships
            self.hunters = {key: val for key, val in state.my_ships.items()
                            if val[0] in self.hunters_pos}
            print(q)
            print(f"{len(self.hunters)} / {len(state.my_ships)}")

        return

    def set_ship_targets(self, state):
        # we choose new targets from a pool of opponent ships. to select the
        # new targets we consider a score composed of "vulnerability" and
        # cargo. to measure vulnerability, we construct a graph where the
        # positions of our hunters have higher weights. the ships that are
        # farthest from their own shipyards are the most vulnerable
        weights = np.ones_like(state.sites)
        if self.hunters_pos.size != 0:
            weights += GRAPH_OPP_WEIGHT * \
                np.sum(state.dist[self.hunters_pos, :] <= 2, axis=0)

        graph = make_graph_csr(state, weights)

        # calculate position, halite, and vulnerability for all opponent ships
        opp_ship_pos = np.array([]).astype(int)
        opp_ship_hal = np.array([]).astype(int)
        opp_ship_vul = np.array([]).astype(int)

        for opp in state.opp_data:
            _hal, yards, ship_pos, ship_hal = opp
            ship_vul = dijkstra(graph, indices=yards, min_only=True)
            ship_vul = ship_vul[ship_pos]
            opp_ship_pos = np.append(opp_ship_pos, ship_pos)
            opp_ship_hal = np.append(opp_ship_hal, ship_hal)
            opp_ship_vul = np.append(opp_ship_vul, ship_vul)

        # if a ship has vulnerability <= 1, it will escape by reaching a yard
        # on this turn or the next, so we remove them from both from the
        # targeting pool and the targets already set
        opp_ship_pos = opp_ship_pos[opp_ship_vul > 1]
        opp_ship_hal = opp_ship_hal[opp_ship_vul > 1]
        opp_ship_vul = opp_ship_vul[opp_ship_vul > 1]

        # store positions of previous targets that are still alive
        prev = np.array([state.opp_ships[key][0] for key in self.ship_targets
                         if key in state.opp_ships]).astype(int)

        # and get the indices of the ships that are already targeted
        # the remaining ships form the pool of possible new targets
        # (note: ~ = not but numpy doesn't like not...)
        boolean_inds = np.in1d(opp_ship_pos, prev)
        target_inds = np.flatnonzero(boolean_inds)
        pool_inds = np.flatnonzero(~boolean_inds)

        # determine how many targets we would like to have
        num_ship_targets = round(len(self.hunters) / HUNTERS_PER_TARGET)

        # and add targets from the pool until the pool is empty
        # or we have enough targets. we add the untargeted ship with the
        # highest score defined as cargo * vulnerability
        opp_ship_score = opp_ship_hal * opp_ship_vul
        opp_ship_score[target_inds] = -1

        # this loop works because we set opp_ship_score < 0 at all selected
        # (target) indices so the argmax always selects a pool_ind where
        # opp_ship_score >= 0
        while target_inds.size < num_ship_targets and pool_inds.size > 0:
            new_ind = np.argmax(opp_ship_score)
            pool_inds = np.setdiff1d(pool_inds, new_ind)
            opp_ship_score[new_ind] = -1

            # only actually set the bounty if there are also some hunters
            # in the area to attack the ship
            near = state.dist[self.hunters_pos, opp_ship_pos[new_ind]] <= 3
            if np.count_nonzero(near) >= 0:
                target_inds = np.append(target_inds, new_ind)
                self.total_bounties += 1  # remove

        # stats - remove eventually
        gotem = np.array([stats.last_state.opp_ships[key][1] for key in
                          self.ship_targets if key not in state.opp_ships])
        self.conversions += gotem.size
        self.total_loot += np.sum(gotem)

        # write the new targets in the ship_targets list
        # and set position/halite/rewards for the targets
        self.ship_targets = [key for key, val in state.opp_ships.items()
                             if val[0] in opp_ship_pos[target_inds]]
        self.ship_targets_pos = opp_ship_pos[target_inds]
        self.ship_targets_hal = opp_ship_hal[target_inds]
        self.ship_targets_rew = 1000 * np.ones_like(self.ship_targets_pos)

        return

    def set_yard_targets(self, state):
        return

    def get_arrays(self, actor):
        positions = np.array([]).astype(int)
        rewards = np.array([]).astype(int)

        if actor in self.hunters:
            pos, hal = self.hunters[actor]
            inds = np.flatnonzero(self.ship_targets_hal > hal)
            positions = np.append(positions, self.ship_targets_pos[inds])
            rewards = np.append(rewards, self.ship_targets_rew[inds])

        return positions, rewards


# ------------ CODE IMPORTED FROM fifos.py ------------ #
class Fifos:
    def __init__(self):
        self.fifo_pos = np.array([]).astype(int)
        self.stripped = {}
        return

    def update(self, state):
        # if we never add any fifo yards, strip() and resolve() have no
        # effect - so if USE_FIFO_SYSTEM is False, update() does nothing
        if not USE_FIFO_SYSTEM:
            return

        # remove any fifo yard positions that may have been destroyed
        self.fifo_pos = np.intersect1d(self.fifo_pos, state.my_yard_pos)

        # mark all yards as fifo yards if opponents
        # are going around attacking shipyards
        if stats.yard_attacks:
            self.fifo_pos = state.my_yard_pos

        return

    def strip(self, state, queue):
        # if x is the position of a fifo yard with a ship on it,
        # self.stripped[x] is the key of the ship
        self.stripped = {val[0]: key for key, val in state.my_ships.items()
                         if val[0] in self.fifo_pos}

        # strip these ships from the scheduling queue
        queue.ships = list(set(queue.ships) - set(self.stripped.values()))

        return

    def resolve(self, state, queue, actor, action):
        # if we spawned at a fifo yard and there is a ship there,
        # schedule the outgoing ship to make room
        if actor in state.my_yards:
            pos = state.my_yards[actor]
            if pos in self.stripped and action == "SPAWN":
                queue.ships.append(self.stripped[pos])

        # same if a ship moves onto a fifo yard and there is a ship there
        if actor in state.my_ships:
            pos, hal = state.my_ships[actor]
            if pos in self.stripped:
                queue.ships.append(self.stripped[pos])

        return


# ------------ CODE IMPORTED FROM decide.py ------------ #
def decide(state, actor):
    # legal moves not resulting in self-collision
    no_self_col = [action for action in state.legal_actions(actor)
                   if not state.self_collision(actor, action)]

    if actor in state.my_ships:
        # check whether we should convert
        if "CONVERT" in no_self_col and should_convert(actor, state):
            decision = "CONVERT"

        # if not, choose a target and head towards it
        else:
            pos, hal = state.my_ships[actor]

            # construct a weighted graph to calculate distances on
            weights = make_weights(actor, state)
            graph = make_graph_csr(state, weights)

            # calculate the distance from all sites to the ship
            # also calculate the distance from all sites to the immediate
            # neighbors of ship - then we can easily take a step "towards"
            # any given site later
            nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]
            hood = np.array([state.newpos(pos, move) for move in nnsew])
            hood_dists = dijkstra(graph, indices=hood)
            ship_dists = hood_dists[nnsew.index(None), :]

            # calculate the distances from all sites to nearest yard
            yard_dists = dijkstra(graph, indices=state.my_yard_pos,
                                  min_only=True)

            # make an effective map of the relative rewards at each site
            effmap = make_map(actor, state)

            # ignore log(0) = -inf warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # the idea is that the expected value of going to any site is
                # the reward at that site, discounted by the time it takes to
                # get to the site and back to the nearest yard
                rate = interest_rate(state, actor)
                values = np.log(hal + effmap) \
                       - np.log((1 + rate)) * (1 + yard_dists + ship_dists)

                # don't want to apply this formula for the yard locations
                # because it causes ships to idle one step away from the yard
                values[state.my_yard_pos] = np.log(hal) \
                    - np.log((1 + rate)) * ship_dists[state.my_yard_pos]

            # the destination is the site with the highest reward
            destination = values.argmax()

            # produce a list of candidate moves and try to decrease
            # distance to destination as much as possible
            candidates = filter_moves(actor, state)
            dist_after = lambda x: hood_dists[nnsew.index(x), destination]
            decision = min(candidates, key=dist_after)

    elif actor in state.my_yards:
        if "SPAWN" in no_self_col and should_spawn(state, actor):
            decision = "SPAWN"
        else:
            decision = None

    return decision


def should_spawn(state, actor=None):
    # check how many ships are right next to the yard. if there are 3 or
    # more (counting fifo ships) we should not spawn until traffic eases
    if actor is not None:
        pos = state.my_yards[actor]
        if np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1) >= 3:
            return False

    # my number of ships (not counting fifo) and score = halite + cargo
    my_ships = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos).size
    my_score = state.my_halite + np.sum(state.my_ship_hal)

    # don't spawn if we have the maximum number of ships
    if my_ships >= MAX_SHIPS:
        return False

    # otherwise we determine a minimum number of ships to keep based on
    # opponent's scores and number of ships
    num_ships = lambda x: x[2].size
    score = lambda x: x[0] + np.sum(x[3])
    max_opp_ships = max([num_ships(opp) for opp in state.opp_data])
    max_opp_score = max([score(opp) for opp in state.opp_data])

    # generally, we want to keep at least as many ships as the opponents
    # so that we don't get completely overrun even when we are trailing
    # but we take this less seriously at the end of the game
    offset = 5 * (state.step / state.total_steps)
    min_ships = max_opp_ships - offset

    # regardless of what opponents do, keep a minimum amount of ships
    min_ships = max(min_ships, MIN_SHIPS)

    # there are two special cases. if we're leading by a lot, we should keep
    # spawning and increase our control of the board. we keep a buffer
    # that ensures we can spawn/convert if something unexpected happens
    # the formula below is buffer = 500 (= spawnCost) halfway through the game
    # and buffer = 1500 (= spawnCost + 2 * convertCost) when there are < 50
    # steps remaining
    op_costs = state.config.spawnCost + state.config.convertCost
    buffer = 2 * op_costs * ((state.step / state.total_steps) ** 2)
    if my_score > max_opp_score + buffer:
        min_ships = MAX_SHIPS

    # the other special case is that we shouldn't spawn too much at the end
    # of the game. these ships won't pay for their cost before the game ends
    if state.total_steps - state.step < STEPS_FINAL:
        min_ships = 5

    # spawn if we have less ships than we need
    return my_ships < min_ships


def should_convert(actor, state):
    # decrease SHIPS_PER_YARD if we have a lot of ships
    ships_per_yard = BASELINE_SHIPS_PER_YARD
    num_ships = len(state.my_ships)
    ships_per_yard += (num_ships > 25) + (num_ships > 35)
    max_yards = round(state.my_ship_pos.size / ships_per_yard)

    # don't build too many yards per ship
    if state.my_yard_pos.size >= max_yards:
        return False

    # keep only a minimal number of yards at the end of the game
    # survival() already guarantees that we always have at least one
    if state.total_steps - state.step < STEPS_FINAL:
        return False

    # get the cluster containing our ship
    pos, hal = state.my_ships[actor]
    cluster = state.get_cluster(pos)

    # don't build a yard if the ship is an outlier
    if cluster.size <= 1:
        return False

    # don't build a yard if the cluster already has a yard
    if np.intersect1d(cluster, state.my_yard_pos).size != 0:
        return False

    # finally only build a yard if we maximize the mean l1 distance
    # to the yards we already own. if this is not the case, the ship
    # in the cluster that does maximize this should convert instead
    mean_dists = np.mean(state.dist[state.my_yard_pos, :], axis=0)
    if mean_dists[pos] < np.max(mean_dists[cluster]):
        return False

    # if none of the above restrictions apply, build a yard
    return True


def filter_moves(ship, state):
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
                  state.opp_collision(ship, move)]

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

    # there are two situations where None is not a good move and
    # we would prefer to remove it from candidates if possible
    if None in candidates and len(candidates) >= 2:
        # don't idle on shipyards - we need room to spawn and deposit
        if pos in state.my_yard_pos:
            candidates.remove(None)

        # if we cannot avoid opponent collisions with certainty, we should
        # move. since hunters will send a ship onto the square we're occupying,
        # we can sometimes get lucky by going to a square no longer covered
        if opp_col_flag:
            candidates.remove(None)

    return candidates


def interest_rate(state, ship):
    pos, hal = state.my_ships[ship]

    # calculate how much space the opponents control with ships that have
    # less halite than we do
    preds = state.opp_ship_pos[state.opp_ship_hal <= hal]
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

    # bring cargo home quicker at the beginning
    if state.step < STEPS_INITIAL:
        risk += 0.01

    # keep a minimum rate of 1 percent even if map is clear
    return 0.02 + risk


def make_map(actor, state):
    # the intrisic value of a site is how much halite we can collect there
    effmap = state.config.collectRate * state.halite_map

    # black out any halite next to opposing team's yards
    # our ships can get stuck there if a ship is camping on the yard
    if state.opp_yard_pos.size != 0:
        yards = state.opp_yard_pos
        hood = np.flatnonzero(np.amin(state.dist[yards, :], axis=0) <= 1)
        effmap[hood] = 0

    # see what targets have been set for our ship and add them to the map
    positions, rewards = targets.get_arrays(actor)
    if positions.size != 0:
        effmap[positions] = rewards

    return effmap


def make_weights(actor, state):
    weights = np.ones_like(state.sites)

    # ships contribute weights in the space they control (which is the ball
    # of radius 1 or 2 around their position). sites controlled by multiple
    # ships should get higher weights

    # heuristic: going "through a site" usually takes two steps. if you want
    # to go "around the site" while staying 1 step away it takes 4 steps so
    # the weight should be > 4/2 = 2
    my_weight = GRAPH_MY_WEIGHT

    # going "through 3 sites" usually takes 4 steps. if you want to go
    # "around the 3 sites" while staying 2 steps from the middle, it takes 8
    # steps so the weight should be > 8/4 = 2. but we want to be very scared
    # of opponent ships so we set this to 4
    opp_weight = GRAPH_OPP_WEIGHT

    pos, hal = state.my_ships[actor]

    # ignore immediate neighbors in the friendly weights - this is handled
    # automatically and the weights can cause traffic jams at close range
    # also ignore weights of any ships on fifo yards
    friendly = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos)
    friendly = friendly[state.dist[pos, friendly] > 1]
    if friendly.size != 0:
        weights += my_weight * np.sum(state.dist[friendly, :] <= 1, axis=0)

    # only consider opponent ships with less halite
    less_hal = state.opp_ship_pos[state.opp_ship_hal <= hal]
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
        self.last_state = None
        self.yard_attacks = False
        return

    def update(self, state):
        if self.last_state is None:
            self.last_state = state
            return

        # check if any of the players lost a yard in the last turn
        # this works because if a yard gets destroyed, it takes at least
        # one turn to move a new ship there before it can convert
        opp_diff = np.setdiff1d(self.last_state.opp_yard_pos,
                                state.opp_yard_pos)
        my_diff = np.setdiff1d(self.last_state.my_yard_pos, state.my_yard_pos)

        if opp_diff.size + my_diff.size > 0:
            self.yard_attacks = True

        self.last_state = state
        return


# ------------ CODE IMPORTED FROM survive.py ------------ #
def survive(state, queue):
    actions = {}

    if len(state.my_yards) == 0:
        # see if it is possible to convert the ship with the most cargo
        # if so, convert it, update state, and return to normal operation
        halite = lambda ship: state.my_ships[ship][1]
        max_cargo = max(queue.ships, key=halite)
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

    while queue.pending():
        # schedule the ship with most halite
        halite = lambda ship: state.my_ships[ship][1]
        ship = max(queue.ships, key=halite)
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

    while queue.pending():
        # schedule the next yard
        yard = queue.yards[0]
        queue.remove(yard)

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


# ------------ CODE IMPORTED FROM state.py ------------ #
class State:
    def __init__(self, obs, config):
        # keep copies of the observation parameters
        self.obs = obs
        self.config = config

        # game configuration
        self.map_size = config.size
        self.total_steps = config.episodeSteps
        self.step = obs.step
        self.halite_map = np.array(obs.halite)

        # my halite, yards, and ships
        my_id = obs.player
        self.my_halite, self.my_yards, self.my_ships = obs.players[my_id]

        # set joint and individual data for oppenents
        self.set_opp_data(obs)

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
        cols = self.sites % self.map_size
        rows = self.sites // self.map_size
        coldist = cols - cols[:, np.newaxis]
        rowdist = rows - rows[:, np.newaxis]
        coldist = np.fmin(np.abs(coldist), self.map_size - np.abs(coldist))
        rowdist = np.fmin(np.abs(rowdist), self.map_size - np.abs(rowdist))
        self.dist = coldist + rowdist

        # sets a number of numpy arrays deriving from self.my_ships, etc
        self.set_derived()

        # compute clusters used for shipyard conversion
        self.make_clusters()
        return

    def set_opp_data(self, obs):
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
    # do this by calling set_derived()
    def set_derived(self):
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

    def make_clusters(self):
        # array of all sites we are currently occupying
        self.cluster_pos = np.append(self.my_ship_pos, self.my_yard_pos)
        self.cluster_pos = np.unique(self.cluster_pos)

        # if there are not enough sites to cluster, set them all as outliers
        if self.cluster_pos.size <= MIN_CLUSTER_SIZE:
            self.cluster_labels = - np.ones_like(self.cluster_pos)
            return

        # create a graph for clustering - see explanation in make_weights()
        my_weight, opp_weight = GRAPH_MY_WEIGHT, GRAPH_OPP_WEIGHT

        weights = np.ones_like(self.sites)

        weights += my_weight * \
            np.sum(self.dist[self.my_ship_pos, :] <= 1, axis=0)

        if self.opp_ship_pos.size != 0:
            weights += opp_weight * \
                np.sum(self.dist[self.opp_ship_pos, :] <= 2, axis=0)

        if self.opp_yard_pos.size != 0:
            weights[self.opp_yard_pos] += opp_weight

        # unlike make_weights() we DON'T want to remove any weights on
        # our yards for this part
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
            return np.array([]).astype(int)

        ind = np.flatnonzero(self.cluster_pos == pos)
        label = self.cluster_labels[ind]

        if label == -1:  # outliers - return only the position
            return np.array([pos])
        else:
            cluster_inds = np.flatnonzero(self.cluster_labels == label)
            return self.cluster_pos[cluster_inds]

#######################################################

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
        # if actor is a yard only spawning has an effect on state
        if (actor in self.my_yards) and (action == "SPAWN"):
            # generate a unique id string
            newid = f"spawn[{actor}]-{str(uuid4())}"

            # create a new ship with no cargo at yard position
            pos = self.my_yards[actor]
            self.my_ships[newid] = [pos, 0]

            # the result is a ship here that cannot move this turn
            self.moved_this_turn = np.append(self.moved_this_turn, pos)

            # subtract spawn cost from available halite
            self.my_halite -= int(self.config.spawnCost)

        if actor in self.my_ships:
            pos, hal = self.my_ships[actor]

            if action == "CONVERT":
                # generate a unique id string
                newid = f"convert[{actor}]-{str(uuid4())}"

                # create a new yard at ship position and remove ship
                self.my_yards[newid] = pos
                del self.my_ships[actor]

                # remove conversion cost from available halite but don't add
                # any net gains - it's not available until next turn
                self.my_halite += min(hal - self.config.convertCost, 0)
                self.halite_map[pos] = 0

            else:
                # if the ship stays put, it can collect halite
                if action is None:
                    collect = self.halite_map[pos] * self.config.collectRate
                    nhal = int(hal + collect)
                    self.halite_map[pos] -= collect
                else:
                    nhal = hal

                # write the new position and halite into my_ships
                npos = self.newpos(pos, action)
                self.my_ships[actor] = [npos, nhal]

                # add position to list of ships that cannot this turn
                self.moved_this_turn = np.append(self.moved_this_turn, npos)

        # update internal variables that derive from my_ships, my_yards
        self.set_derived()
        return

    def legal_actions(self, actor):
        if actor in self.my_yards:
            actions = [None, "SPAWN"]
            # need to have enough halite to spawn
            if self.my_halite < self.config.spawnCost:
                actions.remove("SPAWN")

        if actor in self.my_ships:
            actions = [None, "CONVERT", "NORTH", "SOUTH", "EAST", "WEST"]
            pos, hal = self.my_ships[actor]

            # need to have enough halite. if you only have one ship, you
            # can only convert if you still have enough halite afterwards
            # to spawn a new ship
            minhal = self.config.convertCost - hal
            if len(self.my_ships) == 1:
                minhal += self.config.spawnCost

            # can't convert if you don't have enough halite or are in a yard
            if (self.my_halite < minhal) or (pos in self.my_yard_pos):
                actions.remove("CONVERT")

        return actions

    def self_collision(self, actor, action):
        if actor in self.my_yards:
            # shipyards only give collisions if we spawn a ship after moving
            # a ship there previously
            pos = self.my_yards[actor]
            collision = (action == "SPAWN") and (pos in self.moved_this_turn)

        if actor in self.my_ships:
            pos, hal = self.my_ships[actor]

            # cannot convert onto another shiyard
            if action == "CONVERT":
                collision = pos in self.my_yard_pos
            # otherwise get the new position of the ship and check if
            # it runs into a ship that can no longer move
            else:
                npos = self.newpos(pos, action)
                collision = npos in self.moved_this_turn

        return collision

    def opp_collision(self, ship, action, strict=True):
        # return True if a collision is possible in the next step if ship
        # performs action
        pos, hal = self.my_ships[ship]
        npos = self.newpos(pos, action)

        # check whether we run into a yard
        collision = npos in self.opp_yard_pos

        # find those ships that have less halite than we do
        if strict:
            less_hal = self.opp_ship_pos[self.opp_ship_hal <= hal]
        else:
            less_hal = self.opp_ship_pos[self.opp_ship_hal < hal]

        if less_hal.size != 0:
            # list of sites where ships in less_hal could end up after one turn
            hood = np.flatnonzero(np.amin(self.dist[less_hal, :], axis=0) <= 1)

            # regard our own shipyards as safe havens - this is not necessarily
            # the case but many opponents will not want to run into the yard
            # this will cost them their ship too
            hood = np.setdiff1d(hood, self.my_yard_pos)

            collision = collision or (npos in hood)

        return collision


# ------------ CODE IMPORTED FROM init.py ------------ #
# parameters for graph weighting
GRAPH_MY_WEIGHT = 2
GRAPH_OPP_WEIGHT = 4

# parameters for yard clustering algorithm
MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 5
BASELINE_SHIPS_PER_YARD = 6

# how many steps are the "initial" and "final" phases of the game
STEPS_INITIAL = 50
STEPS_FINAL = 50
STEPS_INTEREST_SPIKE = 15

# parameters for spawning decisions
MAX_SHIPS = 50
MIN_SHIPS = 12

# parameters for setting targets and hunters
HUNTERS_PER_TARGET = 5
MAX_HUNTERS_PER_SHIP = 1

# should we operate the shipyards in FIFO mode?
USE_FIFO_SYSTEM = True

# intialize global strategy object we keep throughout the episode
stats = Stats()
targets = Targets()
fifos = Fifos()


# ------------ CODE IMPORTED FROM agent.py ------------ #
def agent(obs, config):
    # internal game state, to be updated as we decide on actions
    state = State(obs, config)

    # list of ships/yards for which we need to decide on an action
    queue = Queue(state)

    # as we decide on actions for our ships/yards we write them into the
    # actions dictionary, which is what is returned to the environment
    # we initialize actions with the minimal actions needed to ensure
    # survival (usual just the empty dictionary {})
    actions = survive(state, queue)

    # update which yards should operate under fifo system
    # and strip any ships on fifo yards from the queue
    global fifos
    fifos.update(state)
    fifos.strip(state, queue)

    # update any special targets for our ships such as opponent
    # ships/yards that should be targeted by our hunters
    global targets
    targets.update(state)

    # now decide on "normal" actions for the remaining actors
    while queue.pending():
        # schedule the next ship/yard
        actor = queue.schedule(state)

        # decide on an action for it
        action = decide(state, actor)

        # update game state with consequence of action
        state.update(actor, action)

        # put any ships on fifo yards back in the queue if
        # the action resulted in a new ship on a fifo yard
        fifos.resolve(state, queue, actor, action)

        # write action into dictionary of actions to return
        if action is not None:
            actions[actor] = action

    # update the global statistics we track across all turns
    global stats
    stats.update(state)

    print(1 + state.step)
    if state.step == 398:
        print(f"converted {targets.conversions} / {targets.total_bounties}")
        print(f"total: {targets.total_loot}")

    return actions
