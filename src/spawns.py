class Spawns:
    def __init__(self, state, actions):
        # determine how many ships to build
        self.num_ships(state, actions)

        # positions of yards for which we can still decide actions
        yards = [state.my_yards[yard] for yard in actions.yards]
        self.spawn_pos = np.array(yards).astype(int)

        # sort yard positions by preference for spawning - spawn
        # where there are less of our own ships in the area
        inds = np.ix_(state.my_ship_pos, self.spawn_pos)
        traffic = np.sum(state.dist[inds] <= 3, axis=0, initial=0)
        self.spawn_pos = self.spawn_pos[traffic.argsort()]
        return

    def num_ships(self, state, actions):
        # determine how many ships we would like to spawn based on all players'
        # number of ships and score = halite + cargo
        ships = state.my_ship_pos.size
        score = state.my_halite + np.sum(state.my_ship_hal)
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
        if (state.total_steps - state.step < STEPS_FINAL) and (ships >= 5):
            new_ships = 0

        # number of ships wanted without constraints
        self.ships_wanted = int(new_ships)

        # number of ships we can build with constraints
        possible = min(new_ships, state.my_halite // state.spawn_cost)
        self.ships_possible = min(int(possible), len(actions.yards))
        return

    def spawn(self, state, actions):
        # remove spawn_pos that are occupied by ships after this turn
        occupied = state.moved_this_turn[self.spawn_pos]
        free_pos = self.spawn_pos[~occupied]

        # get ids of the yards that should spawn
        pos_to_yard = {v: k for k, v in state.my_yards.items()}
        ids = [pos_to_yard[pos] for pos in free_pos[0:self.ships_possible]]

        # write the appropriate actions into actions.decided
        actions.decided.update({yard: "SPAWN" for yard in ids})
        actions.yards.clear()
        return
