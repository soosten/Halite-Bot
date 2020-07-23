class Queue:
    def __init__(self, state):
        # store ships with list of sites they can go to without colliding
        self.ships = {ship: state.safe_sites(ship) for ship in state.my_ships}

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
        # update the non-colliding mvoes for each ship with
        # the result of the last turn
        self.ships = {key: np.setdiff1d(val, state.moved_this_turn)
                      for key, val in self.ships.items()}

        # first try to schedule any ships with <= 1 possible moves
        stuck = (ship for ship, val in self.ships.items() if val.size <= 1)
        nextup = next(stuck, None)

        # if there are no such ships, choose the one with the most cargo
        if nextup is None:
            cargo = lambda ship: state.my_ships[ship][1]
            nextup = max(self.ships, default=None, key=cargo)

        # if there are no ships, schedule yards by priority
        if nextup is None:
            nextup = max(self.yards, default=None, key=self.yards.get)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup
