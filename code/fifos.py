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

        # mark all yards as fifo yards if opponents
        # are going around attacking shipyards
        if stats.yard_attacks:
            self.fifo_pos = state.my_yard_pos

        return

    def strip(self, state, queue):
        # if x is the position of a fifo yard with a ship on it,
        # self.stripped[x] is the key of the ship
        self.stripped = {val[0]: key for key, val in state.my_ships.items()
                         if val[0] in self.fifo_pos}

        # strip these ships from the scheduling queue
        queue.ships = list(set(queue.ships) - set(self.stripped.values()))

        return

    def resolve(self, state, queue, actor, action):
        # if we spawned at a fifo yard and there is a ship there,
        # schedule the outgoing ship to make room
        if actor in state.my_yards:
            pos = state.my_yards[actor]
            if pos in self.stripped and action == "SPAWN":
                queue.ships.append(self.stripped[pos])

        # same if a ship moves onto a fifo yard and there is a ship there
        if actor in state.my_ships:
            pos, hal = state.my_ships[actor]
            if pos in self.stripped:
                queue.ships.append(self.stripped[pos])

        return
