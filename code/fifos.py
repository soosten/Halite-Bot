class Fifos:
    def __init__(self):
        self.fifo_pos = np.array([]).astype(int)
        self.stripped = {}
        return

    def update(self, state):
        # if we never add any fifo yards, strip() and resolve() have no
        # effect - so if USE_FIFO_SYSTEM is False, update() does nothing
        if not USE_FIFO_SYSTEM:
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
        self.stripped = {val[0]: [key, queue.ships[key]] for key, val in
                         state.my_ships.items() if val[0] in self.fifo_pos}

        # strip these ships from the scheduling queue
        for ship, moves in self.stripped.values():
            queue.remove(ship)

        return

    def resolve(self, state, queue, actor, action):
        # if we spawned at a fifo yard and there is a ship there,
        # schedule the outgoing ship to make room
        if actor in state.my_yards:
            pos = state.my_yards[actor]
            if pos in self.stripped and action == "SPAWN":
                ship, moves = self.stripped[pos]
                queue.ships.update({ship: moves})

        # same if a ship moves onto a fifo yard and there is a ship there
        if actor in state.my_ships:
            pos, hal = state.my_ships[actor]
            if pos in self.stripped:
                ship, moves = self.stripped[pos]
                queue.ships.update({ship: moves})

        return
