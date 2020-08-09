class Fifos:
    def __init__(self):
        self.fifo_pos = np.array([]).astype(int)
        self.stripped = {}
        return

    def update(self, state):
        # if we never add any fifo yards, strip() and resolve() have no
        # effect - so if USE_FIFO_SYSTEM is False, update() does nothing
        if not FIFO_MODE:
            return

        # remove any fifo yard positions that may have been destroyed
        self.fifo_pos = np.intersect1d(self.fifo_pos, state.my_yard_pos)

        # if an opponent that has attacked shipyards in the past gets within
        # radius 2 of a yard, we add the yard to the fifo yards
        attacker_pos = np.array([]).astype(int)
        for opp in stats.yard_attackers:
            attacker_pos = np.append(attacker_pos, state.opp_data[opp][2])

        if attacker_pos.size != 0:
            dist = state.dist[np.ix_(attacker_pos, state.my_yard_pos)]
            inds = np.amin(dist, axis=0) <= 2
            self.fifo_pos = np.union1d(self.fifo_pos, state.my_yard_pos[inds])

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

    def resolve(self, state, queue, actor, action):
        # find out whether the action requires any fifo updates
        if actor in state.my_yards and action == "SPAWN":
            pos = state.my_yards[actor]
        elif actor in state.my_ships:
            pos = state.my_ships[actor][0]
        else:
            pos = -1

        ship, moves = self.stripped.get(pos, [None, None])

        # if ship is None, the actor was not on a fifo position or a
        # yard that didn't spawn so nothing to do here
        if ship is None:
            return

        # otherwise insert the new ship into the queue
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
