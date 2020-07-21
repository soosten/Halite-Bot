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
            candidates = filter_moves(actor, state, destination)
            dist_after = lambda x: hood_dists[nnsew.index(x), destination]
            decision = min(candidates, key=dist_after)

    elif actor in state.my_yards:
        if "SPAWN" in no_self_col and should_spawn(state, actor):
            decision = "SPAWN"
        else:
            decision = None

    return decision


def should_spawn(state, actor=None):
    if actor is not None:
        # check how many ships are right next to the yard. if there are 3 or
        # more (counting fifo ships) we should not spawn until traffic eases
        pos = state.my_yards[actor]
        if np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1) >= 3:
            return False

        # then check if the yard is a fifo yard without a ship
        # and spawn if this is the case
        if pos in fifos.fifo_pos and pos not in state.my_ship_pos:
            return True

    # my number of ships (not counting fifo) and score = halite + cargo
    my_ships = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos).size
    my_score = state.my_halite + np.sum(state.my_ship_hal)

    # determine a number of ships to keep based on opponent's scores and ships
    num_ships = lambda x: x[2].size
    alive = lambda x: (x[1].size + x[2].size) > 0
    score = lambda x: alive(x) * (x[0] + np.sum(x[3]))
    max_opp_ships = max([num_ships(opp) for opp in state.opp_data.values()])
    max_opp_score = max([score(opp) for opp in state.opp_data.values()])

    # don't spawn if we have the maximum number of ships
    # max_ships = min(MAX_SHIPS, max_opp_ships + 15)
    if my_ships >= MAX_SHIPS:
        return False

    # generally, we want to keep at least as many ships as the opponents
    # so that we don't get completely overrun even when we are trailing
    # but we take this less seriously at the end of the game
    offset = 5 * (state.step / state.total_steps)
    min_ships = max_opp_ships - offset

    # regardless of what opponents do, keep a minimum amount of ships
    min_ships = max(min_ships, MIN_SHIPS)

    # there are three special cases:
    # we should always spawn at the beginning of the game
    if state.step < STEPS_INITIAL:
        min_ships = MAX_SHIPS

    # if we're leading by a lot, we should keep spawning and increase our
    # control of the board. we keep a buffer that ensures we can spawn/convert
    # if something unexpected happens. the formula below is buffer = 500
    # (= spawnCost) halfway through the game and buffer = 1500
    # (= spawnCost + 2 * convertCost) when there are < 50 steps remaining
    op_costs = state.config.spawnCost + state.config.convertCost
    buffer = 2 * op_costs * ((state.step / state.total_steps) ** 2)
    if my_score > max_opp_score + buffer:
        min_ships = MAX_SHIPS

    # we shouldn't spawn too much at the end of the game. these ships won't
    # pay for their cost before the game ends
    if state.total_steps - state.step < STEPS_FINAL:
        min_ships = 5

    # spawn if we have less ships than we need
    return my_ships < min_ships


def should_convert(actor, state):
    # decrease SHIPS_PER_YARD if we have a lot of ships
    ships_per_yard = BASELINE_SHIPS_PER_YARD
    num_ships = len(state.my_ships)
    ships_per_yard += (num_ships > 25) + (num_ships > 35)
    max_yards = state.my_ship_pos.size // ships_per_yard

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


def filter_moves(ship, state, destination):
    pos, hal = state.my_ships[ship]

    legal = state.legal_actions(ship)

    # conversion was already considered
    if "CONVERT" in legal:
        legal.remove("CONVERT")

    # no self collisions
    no_self_col = [move for move in legal if not
                   state.self_collision(ship, move)]

    # determine what yards parameter to pass to opp_collision. if our final
    # destination is to destroy a shipyard, we don't check for collision with
    # enemy shipyards
    yards = (destination not in state.opp_yard_pos)

    # if we have a lot of friendly ships nearby its likely that there is a
    # big reward close, so we are more aggressive and set strict = False by
    # default. this should also help clear up traffic jams if a ship is idling
    # next to a big reward. if our goal is to destroy a yard, we shouldn't
    # run away from other ships and also set strict = False.
    friends = np.sum(state.dist[pos, state.my_ship_pos] <= 2)
    strict = (friends < 4) and yards

    # usually strong_no_opp_col has moves that don't result in collision with
    # an enemy ship or yard (unless we set a different default above)
    strong_no_opp_col = [move for move in legal if not
                         state.opp_collision(ship, move, strict, yards)]

    # moves which don't result in collision with an enemy ship
    # that has strictly less cargo
    weak_no_opp_col = [move for move in legal if not
                       state.opp_collision(ship, move, False, yards)]

    # ideally consider moves that don't collide at all
    candidates = list(set(no_self_col) & set(strong_no_opp_col))
    opp_col_flag = False

    # if this is impossible, try weakened no_opp_col
    if len(candidates) == 0:
        candidates = list(set(no_self_col) & set(weak_no_opp_col))

    # if this is impossible, don't collide with opponent ships
    if len(candidates) == 0:
        candidates = strong_no_opp_col

    # if this is impossible, don't collide with opponent ships (weak version)
    if len(candidates) == 0:
        candidates = weak_no_opp_col

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

    # bring cargo home quicker if we need to spawn
    risk += SPAWN_PREMIUM * should_spawn(state)

    # keep a minimum rate of 1 percent even if map is clear
    return BASELINE_INTEREST + risk


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
    positions, rewards = targets.get_arrays(actor, state)
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
