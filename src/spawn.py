def spawn(state, actions):
    # check how many new ships we should build
    new_ships = num_spawns(state)
    new_ships = min(new_ships, state.my_halite // state.spawn_cost)
    new_ships = min(new_ships, len(actions.yards))

    # rank yards by where we should spawn first
    def priority(yard):
        pos = state.my_yards[yard]
        return -np.sum(state.dist[state.my_ship_pos, pos] <= 3)

    while new_ships > 0 and len(actions.yards) > 0:
        yard = max(actions.yards, key=priority)
        actions.yards.remove(yard)
        if not state.moved_this_turn[state.my_yards[yard]]:
            actions.decided[yard] = "SPAWN"
            new_ships -= 1

    actions.yards.clear()

    return


def num_spawns(state):
    # determine how many ships we would like to spawn based on all players'
    # number of ships and score = halite + cargo
    ships = state.my_ship_pos.size
    halite = state.my_halite
    score = halite + memory.cargo
    max_opp_ships = max(state.opp_num_ships.values())
    max_opp_score = max(state.opp_scores.values())

    # keep at least MIN_SHIPS ships
    bound = MIN_SHIPS - ships
    new_ships = max(bound, 0)

    # spawn if we have fewer ships than the opponents
    # but take this less seriously at the end of the game
    offset = SPAWNING_OFFSET * (state.step / state.total_steps)
    bound = max_opp_ships - offset - ships
    new_ships = max(bound, new_ships)

    # spawn if we have more halite than opponents. the buffer below is
    # ~500 (= spawnCost) halfway through the game and ~1500
    # (= spawnCost + 2 * convertCost) when there are < 50 steps remaining
    op_costs = state.spawn_cost + state.convert_cost
    buffer = 2 * op_costs * ((state.step / state.total_steps) ** 2)
    bound = (score - max_opp_score - buffer) // state.spawn_cost
    new_ships = max(bound, new_ships)

    # spawn if we there is a lot of time left
    bound = YARD_SCHEDULE.size * (state.step < SPAWNING_STEP)
    new_ships = max(bound, new_ships)

    # don't spawn if its no longer worth it and we have a few ships
    if (state.total_steps - state.step) < STEPS_FINAL and ships >= 5:
        new_ships = 0

    return new_ships
