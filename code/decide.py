def decide(state, actor):
    # if the actor is a ship, we choose the move with the highest
    # ranking among those that don't result in unfavorable collisions
    if actor in state.my_ships:
        destination = targets.destinations[actor]
        ranking = targets.rankings[actor]
        candidates = filter_moves(actor, state, destination)
        decision = min(candidates, key=lambda move: ranking.index(move))

        # if we decide None, add 1 to the counter of successive idling
        # moves for the ship, otherwise set counter to zero
        idle_count = stats.idle_ships.get(actor, 0)
        stats.idle_ships[actor] = (idle_count + 1) * (decision is None)

    # if the actor is a yard, we check whether we should spawn or not
    elif actor in state.my_yards:
        decision = "SPAWN" if should_spawn(state, actor) else None

    return decision


def should_spawn(state, actor=None):
    if actor is not None:
        # check if spawning is a legal move that doesn't result in
        # self-collision and return False if not
        no_self_col = [action for action in state.legal_actions(actor)
                       if not state.self_collision(actor, action)]
        if "SPAWN" not in no_self_col:
            return False

        # check how many ships are right next to the yard. if there are 4 or
        # more (counting fifo ships) we should not spawn until traffic eases
        pos = state.my_yards[actor]
        if np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1) >= 4:
            return False

        # then check if the yard is a fifo yard without a ship
        # and spawn if this is the case
        if pos in fifos.fifo_pos and pos not in state.my_ship_pos:
            if state.total_steps - state.step > STEPS_FINAL:
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


def filter_moves(ship, state, destination):
    pos, hal = state.my_ships[ship]

    nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]

    # no self collisions
    no_self_col = [move for move in nnsew if not
                   state.self_collision(ship, move)]

    # determine what yards parameter to pass to opp_collision. if our final
    # destination is to destroy a shipyard, we don't check for collision with
    # enemy shipyards
    yards = (destination not in bounties.yard_targets_pos)

    idle_count = stats.idle_ships.get(ship, 0)
    strict = idle_count <= 5

    # usually strong_no_opp_col has moves that don't result in collision with
    # an enemy ship or yard (unless we set a different default above)
    no_opp_col = [move for move in nnsew if not
                  state.opp_collision(ship, move, strict, yards)]

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
        candidates = nnsew

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
