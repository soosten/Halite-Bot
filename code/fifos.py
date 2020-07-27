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
        queue.ships.update({ship: moves})

        # and compute a heuristic target data for ship
        # see the comments in targets.calculate()
        targets.ship_list.append(ship)
        targets.distances[ship] = targets.calc_distances(ship, state)
        dupes, no_dupes = targets.calc_rewards(ship, state, unappended=True)
        targets.rewards[ship] = dupes

        # we choose a destination that has not yet been selected by the
        # optimal assignment algorithm for the non-fifo ships
        taken = np.array(list(targets.destinations.values()))
        no_dupes[taken] = 0
        targets.destinations[ship] = no_dupes.argmax()
        targets.values[ship] = no_dupes.max()

        # and produce a ranking of moves depending on how much
        # they decrease the distance to our new target
        hood_dists = targets.distances[ship][0]
        dest_dists = hood_dists[:, targets.destinations[ship]]
        dist_after = lambda x: dest_dists[targets.nnsew.index(x)]
        targets.rankings[ship] = targets.nnsew.copy()
        targets.rankings[ship].sort(key=dist_after)

        return
