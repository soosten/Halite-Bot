class Queue:
    def __init__(self, state):
        # store ships with list of sites they cannot go to without colliding
        self.ships = {ship: state.unsafe_sites(ship)
                      for ship in state.my_ships}

        # ships that are being hunted
        pos = lambda ship: state.my_ships[ship][0]
        self.hunted = {ship for ship, unsafe in self.ships.items()
                       if unsafe[pos(ship)]}

        # ships with no cargo
        cargo = lambda ship: state.my_ships[ship][1]
        self.empty = {ship for ship in self.ships if cargo(ship) == 0}

        # we can compute the priorities of yards right away we want to spawn
        # first at yards in "contested" areas with lots of opponent ships
        self.yards = {yard: np.sum(state.dist[pos, state.opp_ship_pos] <= 4)
                      for yard, pos in state.my_yards.items()}

        return

    def pending(self):
        return (len(self.ships) + len(self.yards)) > 0

    def remove(self, actor):
        self.ships.pop(actor, None)
        self.yards.pop(actor, None)
        self.hunted.discard(actor)
        self.empty.discard(actor)
        return

    def value(self, ship):
        return targets.values.get(ship, 0)

    def schedule(self, state):
        # update the non-colliding moves for each ship
        self.ships = {key: val | state.moved_this_turn
                      for key, val in self.ships.items()}

        # schedule any ships with <= 1 possible moves
        stuck = (ship for ship, val in self.ships.items() if np.sum(~val) <= 1)
        nextup = next(stuck, None)

        # schedule any ships with opponents in pursuit
        if nextup is None:
            nextup = next(iter(self.hunted), None)

        # schedule any ships with no cargo
        if nextup is None:
            nextup = next(iter(self.empty), None)

        # schedule remaining ships by value
        if nextup is None:
            nextup = max(self.ships, default=None, key=self.value)

        # schedule yards
        if nextup is None:
            nextup = max(self.yards, default=None, key=self.yards.get)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup
