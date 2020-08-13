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
    # my number of ships (not counting fifo) and score = halite + cargo
    my_ships = state.my_ship_pos.size - fifos.fifo_pos.size
    my_score = state.my_halite + np.sum(state.my_ship_hal)

    # check for some special scenarios in which we should not spawn
    if actor is not None:
        # if spawning is not legal
        if "SPAWN" not in state.legal_actions(actor):
            return False

        # if spawing results in collision
        if state.self_collision(actor, "SPAWN"):
            return False

        # if there is too much traffic at the yard
        pos = state.my_yards[actor]
        if np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1) >= 4:
            return False

    # if we already have a lot of ships
    if my_ships >= MAX_SHIPS:
        return False

    # if its the end of the game and we have a few ships still
    endgame = state.total_steps - state.step < STEPS_FINAL
    if endgame and my_ships >= 5:
        return False

    # otherwise the rule is that we spawn in any of the following cases:
    # we have less ships than the opponents, we are leading by a lot,
    # there are still many steps left in the game
    # determine the opponent scores and number of ships
    num_ships = lambda x: x[2].size
    alive = lambda x: (x[1].size + x[2].size) > 0
    score = lambda x: alive(x) * (x[0] + np.sum(x[3]))
    max_opp_ships = max([num_ships(opp) for opp in state.opp_data.values()])
    max_opp_score = max([score(opp) for opp in state.opp_data.values()])

    # spawn if we there is a lot of time left
    spawn = state.step < SPAWNING_STEP

    # spawn if we have fewer ships than the opponents
    # but we take this less seriously at the end of the game
    offset = SPAWNING_OFFSET * (state.step / state.total_steps)
    spawn = spawn or (my_ships < max_opp_ships - offset)

    # spawn if we are below the minimum number of ships
    spawn = spawn or (my_ships < MIN_SHIPS)

    # spawn if we have more halite than opponents. the formula below is
    #  buffer = 500 (= spawnCost) halfway through the game and buffer = 1500
    # (= spawnCost + 2 * convertCost) when there are < 50 steps remaining
    op_costs = state.config.spawnCost + state.config.convertCost
    buffer = 2 * op_costs * ((state.step / state.total_steps) ** 2)
    spawn = spawn or (my_score > max_opp_score + buffer)

    # finally, spawn if the yard is a fifo yard without a ship
    if actor is not None:
        pos = state.my_yards[actor]
        fifo_spawn = (pos in fifos.fifo_pos)
        fifo_spawn = fifo_spawn and (pos not in state.my_ship_pos)
        spawn = spawn or fifo_spawn

    return spawn


def filter_moves(ship, state, destination):
    pos, hal = state.my_ships[ship]

    # determine what yards parameter to pass to opp_collision. if our final
    # destination is to destroy a shipyard, we don't check for collision with
    # enemy shipyards
    yards = (destination not in bounties.yard_targets_pos)

    # relax the strict opponent collision checking if we
    # have been idling at some site
    idle_count = stats.idle_ships.get(ship, 0)
    strict = idle_count <= 5

    # legal moves
    nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]

    # no self collisions
    no_self_col = [move for move in nnsew if not
                   state.self_collision(ship, move)]

    # no undesired opponent collisions
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
