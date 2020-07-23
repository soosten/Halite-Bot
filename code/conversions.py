class Conversions:
    def __init__(self):
        self.yard_scores = None
        return

    def __call__(self, state, queue, actions):
        # first check if we want to build a yard at all
        if not self.need_yard(state):
            return

        # yard_dist = lambda s: np.amin(state.dist[state.my_yard_pos, s[0]])
        # hood = lambda s: np.flatnonzero(state.dist[s[0], :] <= 4)
        # score = lambda s: np.sum(state.halite_map[hood(s)] > 0)

        def yard_dist(ship):
            dist = state.dist[state.my_yard_pos, state.my_ships[ship][0]]
            return np.amin(dist)

        def score(ship):
            hood = state.dist[state.my_ships[ship][0], :] <= HALITE_RADIUS
            return np.sum(state.halite_map[hood] > 0)

        candidates = {ship: score(ship) for ship in queue.ships.keys()
                      if yard_dist(ship) >= MIN_YARD_DIST
                      and score(ship) >= MIN_HALITE_CELLS}

        convert = max(candidates, default=None, key=candidates.get)

        if convert is not None:
            if "CONVERT" in state.legal_actions(convert):
                print(f"converting score = {candidates[convert]}")
                queue.remove(convert)
                state.update(convert, "CONVERT")
                actions[convert] = "CONVERT"

        return

    def need_yard(self, state):
        # number of ships without fifo ships
        num_ships = state.my_ship_pos.size - fifos.fifo_pos.size

        # increase SHIPS_PER_YARD if we have a lot of ships
        ships_per_yard = BASELINE_SHIPS_PER_YARD
        ships_per_yard += (num_ships > 25) + (num_ships > 35)

        # number of yards to build
        num_yards = num_ships // ships_per_yard

        # keep at least 2 yards until the final steps
        if state.total_steps - state.step > STEPS_FINAL:
            num_yards = max(num_yards, 2)

        # build a yard if we have < num_yards yards
        return state.my_yard_pos.size < num_yards
