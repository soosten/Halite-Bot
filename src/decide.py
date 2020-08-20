def decide(state, actor):
    # if the actor is a ship, we choose the move with the highest
    # ranking among those that don't result in unfavorable collisions
    if actor in state.my_ships:
        dest = targets.destinations[actor]
        ranking = targets.rankings[actor]
        candidates = filter_moves(actor, state, dest)
        decision = min(candidates, key=lambda move: ranking.index(move))

    # if the actor is a yard, we check whether we should spawn or not
    elif actor in state.my_yards:
        decision = "SPAWN" if should_spawn(state, actor) else None

    return decision


def should_spawn(state, yard=None):
    # my number of ships (not counting fifo) and score = halite + cargo
    my_ships = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos).size
    my_score = state.my_halite + np.sum(state.my_ship_hal)
    max_opp_ships = max(state.opp_num_ships.values())
    max_opp_score = max(state.opp_scores.values())

    # check for some special scenarios in which we should not spawn
    if yard is not None:
        illegal = "SPAWN" not in state.legal_actions(yard)
        collision = state.self_collision(yard, "SPAWN")

        pos = state.my_yards[yard]
        traffic_jam = np.sum(state.dist[state.my_ship_pos, pos] <= 1) >= 4

        if illegal or collision or traffic_jam:
            return False

    # if we already have enough ships
    enough = MAX_SHIPS if (state.total_steps - state.step) > STEPS_FINAL else 5
    if my_ships >= enough:
        return False

    # spawn if we there is a lot of time left
    spawn = state.step < SPAWNING_STEP

    # spawn if we have fewer ships than the opponents
    # but take this less seriously at the end of the game
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
    if yard is not None:
        pos = state.my_yards[yard]
        fifo_spawn = (pos in fifos.fifo_pos)
        fifo_spawn = fifo_spawn and (pos not in state.my_ship_pos)
        spawn = spawn or fifo_spawn

    return spawn


def filter_moves(ship, state, dest):
    pos, hal = state.my_ships[ship]

    # there are some situations where None is not a good move and
    # we would prefer to remove it from candidates if possible
    none_flag = False

    # legal moves
    nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]

    # no self collisions
    no_self_col = [move for move in nnsew if not
                   state.self_collision(ship, move)]

    # no undesired opponent collisions - check strictly if the ship has
    # cargo or is far from the yard. hopefully, this will make ships less
    # locked down if some attackers are waiting in front of our yard
    strict = np.amin(state.dist[state.my_yard_pos, pos], axis=0) > 2
    strict = strict or (hal > 0)
    no_opp_col = [move for move in nnsew if not
                  state.opp_collision(ship, move, strict)]

    # turn off self collision checking if we are depositing at the end
    if state.total_steps - state.step < STEPS_SPIKE:
        yard_ind = np.argmin(state.dist[state.my_yard_pos, pos], axis=0)
        yard = state.my_yard_pos[yard_ind]
        close = state.dist[yard, pos] <= 3
        traffic = np.sum(state.dist[state.my_ship_pos, yard] <= 4) >= 10
        if traffic and close:
            no_self_col = nnsew

    # turn off opponent collision checking if we are hunting a yard
    # and are close enough to strike
    if (dest in state.opp_yard_pos) and (state.dist[pos, dest] <= 2):
        no_opp_col = nnsew

    # ideally consider moves that don't collide at all
    candidates = list(set(no_self_col) & set(no_opp_col))

    # if this is impossible, don't collide with opponent ships
    if len(candidates) == 0:
        candidates = no_opp_col

    # if this is impossible, don't collide with own ships
    if len(candidates) == 0:
        # don't stay put if we are being hunted - could get lucky by moving
        none_flag = True
        candidates = no_self_col

    # if this is impossible, make a legal move
    if len(candidates) == 0:
        candidates = nnsew

    # don't pick up unwanted cargo that makes ships vulnerable
    no_cargo = (pos != dest)
    no_cargo = no_cargo and (state.halite_map[pos] > 0)
    no_cargo = no_cargo and (dest not in state.my_yard_pos)
    none_flag = none_flag or no_cargo

    if none_flag and (None in candidates) and (len(candidates) >= 2):
        candidates.remove(None)

    return candidates
