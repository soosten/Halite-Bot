class Conversions:
    def __init__(self):
        self.yard_scores = None
        return

    def __call__(self, state, queue, actions):
        # first check if we want to build a yard at all
        if not self.need_yard(state):
            return

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

        if convert is not None and "CONVERT" in state.legal_actions(convert):
            queue.remove(convert)
            state.update(convert, "CONVERT")
            actions[convert] = "CONVERT"

        return

    def need_yard(self, state):
        # don't build any yards in the final phase of the game
        # survive() ensures we keep at least one
        if state.total_steps - state.step < STEPS_FINAL:
            return False

        # number of ships without fifo ships
        num_ships = state.my_ship_pos.size - fifos.fifo_pos.size

        num_yards = 3 if num_ships > 20 else 2

        # build a yard if we have < num_yards yards
        return state.my_yard_pos.size < num_yards
