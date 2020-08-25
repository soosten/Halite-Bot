class Spawns:
    def __init__(self, state, actions):
        # rank yards by where we should spawn first
        def priority(yard):
            pos = state.my_yards[yard]
            return -np.sum(state.dist[state.my_ship_pos, pos] <= 3)

        self.yards = {yard: priority(yard) for yard in actions.yards}

        # determine how many ships we would like to spawn based on all players'
        # number of ships and score = halite + cargo
        ships = state.my_ship_pos.size
        halite = state.my_halite
        score = halite + np.sum(state.my_ship_hal)
        max_opp_ships = max(state.opp_num_ships.values())
        max_opp_score = max(state.opp_scores.values())

        new_ships = 0

        # keep at least MIN_SHIPS ships
        bound = MIN_SHIPS - ships
        new_ships = max(bound, new_ships)

        # spawn if we have fewer ships than the opponents
        # but take this less seriously at the end of the game
        offset = SPAWNING_OFFSET * (state.step / state.total_steps)
        bound = max_opp_ships - offset - ships
        new_ships = max(bound, new_ships)

        # spawn if we have more halite than opponents. the buffer below is
        # ~500 (= spawnCost) halfway through the game and ~1500
        # (= spawnCost + 2 * convertCost) when there are < 50 steps remaining
        op_costs = state.config.spawnCost + state.config.convertCost
        buffer = 2 * op_costs * ((state.step / state.total_steps) ** 2)
        bound = (score - max_opp_score - buffer) // state.config.spawnCost
        new_ships = max(bound, new_ships)

        # spawn if we there is a lot of time left
        bound = len(actions.yards) * (state.step < SPAWNING_STEP)
        new_ships = max(bound, new_ships)

        # don't spawn if its no longer worth it and we have a few ships
        if (state.total_steps - state.step) < STEPS_FINAL and ships >= 5:
            new_ships = 0

        # can't spawn more than we can afford or have yards for
        new_ships = min(new_ships, halite // state.config.spawnCost)
        self.num_spawns = min(new_ships, len(actions.yards))

        return
