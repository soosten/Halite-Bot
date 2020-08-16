def survive(state, queue):
    actions = {}

    if len(state.my_yards) == 0:
        # see if it is possible to convert the ship with the most cargo
        # if so, convert it, update state, and return to normal operation
        cargo = lambda ship: state.my_ships[ship][1]
        max_cargo = max(queue.ships.keys(), key=cargo)

        if "CONVERT" in state.legal_actions(max_cargo):
            queue.remove(max_cargo)
            actions[max_cargo] = "CONVERT"
            state.update(max_cargo, "CONVERT")

        # otherwise instruct all ships to survive as long as possible while
        # collecting as much halite as possible
        else:
            actions = run_and_collect(state, queue)

    # elif since it can never happen that we start the turn without any
    # ships or yards - but it can happen that the previous block converted
    # our last ship
    elif len(state.my_ships) == 0:
        # spawn as many ships as we can afford
        actions = spawn_maximum(state, queue)

    return actions


def run_and_collect(state, queue):
    actions = {}

    while queue.pending():
        ship = queue.schedule(state)
        pos, hal = state.my_ships[ship]

        # legal moves
        nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]

        # no self collisions
        no_self_col = [move for move in nnsew if not
                       state.self_collision(ship, move)]

        # no undesired opponent collisions
        no_opp_col = [move for move in nnsew if not
                      state.opp_collision(ship, move)]

        # ideally consider moves that don't collide at all
        candidates = list(set(no_self_col) & set(no_opp_col))

        # if this is impossible, don't collide with opponent ships
        if len(candidates) == 0:
            candidates = no_opp_col

        # if this is impossible, don't collide with own ships
        if len(candidates) == 0:
            candidates = no_self_col

        # if this is impossible, make a legal move
        if len(candidates) == 0:
            candidates = nnsew

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
        yard = queue.schedule(state)

        # if spawning is legal, do it
        if "SPAWN" in state.legal_actions(yard):
            state.update(yard, "SPAWN")
            actions[yard] = "SPAWN"

    return actions
