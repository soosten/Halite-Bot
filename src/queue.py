class Queue:
    def __init__(self, state):
        # store ships with list of sites they cannot go to without colliding
        self.ships = {ship: state.unsafe_sites(ship)
                      for ship in state.my_ships}

        # store positions of those ships that are being pursued by opponents
        near = state.dist[np.ix_(state.my_ship_pos, state.opp_ship_pos)] <= 1
        threats = state.my_ship_hal[:, np.newaxis] > state.opp_ship_hal
        hunters = np.count_nonzero(near & threats, axis=1)
        self.hunted = state.my_ship_pos[hunters > 0]

        # we can compute the priorities of yards right away
        # we want to spawn first a yard that are in "contested" areas
        # where there are lots of opponent ships. we give highest
        # priority to those that have a ship right next to them
        self.yards = {}
        for yard, pos in state.my_yards.items():
            dist1 = np.sum(state.dist[pos, state.opp_ship_pos] <= 1)
            dist4 = np.sum(state.dist[pos, state.opp_ship_pos] <= 4)
            self.yards[yard] = 100 * dist1 + dist4

        return

    def pending(self):
        return (len(self.ships) + len(self.yards)) > 0

    def remove(self, actor):
        self.ships.pop(actor, None)
        self.yards.pop(actor, None)
        return

    def schedule(self, state):
        # update the non-colliding moves for each ship with
        # the result of the last turn
        self.ships = {key: val | state.moved_this_turn
                      for key, val in self.ships.items()}

        # first schedule any ships with <= 1 possible moves
        stuck = (ship for ship, val in self.ships.items() if np.sum(~val) <= 1)
        nextup = next(stuck, None)

        # then schedule any ships with opponents in pursuit
        if nextup is None:
            pos = lambda ship: state.my_ships[ship][0]
            running = (ship for ship in self.ships if pos(ship) in self.hunted)
            nextup = next(running, None)

        # then schedule any ships with no cargo
        if nextup is None:
            cargo = lambda ship: state.my_ships[ship][1]
            empty = (ship for ship in self.ships if cargo(ship) == 0)
            nextup = next(empty, None)

        # if there are no such ships, choose the one with the highest value
        if nextup is None:
            value = lambda ship: targets.values.get(ship, 0)
            nextup = max(self.ships, default=None, key=value)

        # if there are no ships, schedule yards by priority
        if nextup is None:
            nextup = max(self.yards, default=None, key=self.yards.get)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup
