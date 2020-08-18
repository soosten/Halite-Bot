class Fifos:
    def __init__(self):
        self.fifo_pos = np.array([]).astype(int)
        self.stripped = {}
        return

    def update(self, state):
        # if we never add any fifo yards, strip() and resolve() have no
        # effect - so if FIFO_MODE is False, update() does nothing
        if not FIFO_MODE:
            return

        # remove any fifo yard positions that may have been destroyed
        self.fifo_pos = np.intersect1d(self.fifo_pos, state.my_yard_pos)

        # if there is an opponent ship within distance 2 of the yard,
        # add this yard to fifo yards
        if state.opp_ship_pos.size != 0:
            dist = state.dist[np.ix_(state.opp_ship_pos, state.my_yard_pos)]
            yards = state.my_yard_pos[np.amin(dist, axis=0) <= 2]
            self.fifo_pos = np.union1d(self.fifo_pos, yards)

        return

    def strip(self, state, queue):
        # if x is the position of a fifo yard with a ship on it,
        # self.stripped[x] is the [key, safe_sites] of the ship
        pos = lambda ship: state.my_ships[ship][0]
        self.stripped = {pos(ship): [ship, val] for ship, val in
                         queue.ships.items() if pos(ship) in self.fifo_pos}

        # strip these ships from the scheduling queue
        for ship, moves in self.stripped.values():
            queue.remove(ship)

        return

    def resolve(self, state, queue):
        # fifo ships that need to move
        res_pos = [pos for pos in self.stripped if state.moved_this_turn[pos]]

        for pos in res_pos:
            # move fifo ship from stripped into the queue
            ship, moves = self.stripped.pop(pos)
            queue.ships[ship] = moves

            # we choose a destination that has not yet been selected by the
            # optimal assignment algorithm for the non-fifo ships
            targets.ship_list.append(ship)
            targets.distances[ship] = targets.calc_distances(ship, state)
            rewards = targets.calc_rewards(ship, state, return_appended=False)
            taken = np.array(list(targets.destinations.values())).astype(int)
            rewards[taken] = 0
            targets.add_destination(ship, rewards.argmax(), rewards.max())

        return
