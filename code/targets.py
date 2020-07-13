class Targets:
    def __init__(self):
        self.hunters = {}
        self.hunters_pos = np.array([]).astype(int)
        self.hunters_hal = np.array([]).astype(int)

        self.ship_targets = []
        self.ship_targets_pos = np.array([]).astype(int)
        self.ship_targets_hal = np.array([]).astype(int)
        self.ship_targets_rew = np.array([]).astype(int)

        # remove
        self.total_bounties = 0
        self.conversions = 0
        self.total_loot = 0

        return

    def update(self, state):
        self.set_hunters(state)
        self.set_ship_targets(state)
        self.set_yard_targets(state)
        return

    def set_hunters(self, state):
        # determine hunters depending on cargo and halite on the map
        if state.my_ship_pos.size == 0:
            self.hunters = {}
        else:
            # determine a mining value of each ship based on its cargo
            # and the amount of halite near it
            # for each ship position x, hood_hal[x] is the average halite in
            # the cells around x. we add this to the cargo to break ties in
            # values of ships with equal cargo
            boolhood = state.dist[state.my_ship_pos, :] <= 2
            hood = np.apply_along_axis(np.flatnonzero, 1, boolhood)
            hood_hal = np.mean(state.halite_map[hood], axis=1)
            ship_val = state.my_ship_hal + 0.1 * hood_hal

            # choose a fraction of ships depending on halite on the map
            q = MAX_HUNTERS_PER_SHIP
            q -= np.sum(state.halite_map) / state.config.startingHalite
            q = q if q > 0.3 else 0
            num_hunters = int(q * len(state.my_ships))

            # designate the num_hunters ships with lowest value as hunters
            hunters_ind = np.argpartition(ship_val, num_hunters)
            hunters_ind = hunters_ind[0:num_hunters]
            self.hunters_pos = state.my_ship_pos[hunters_ind]

            # remove any fifo ships from the hunters
            self.hunters_pos = np.setdiff1d(self.hunters_pos, fifos.fifo_pos)

            # also keep hunters in dictionary form like state.my_ships
            self.hunters = {key: val for key, val in state.my_ships.items()
                            if val[0] in self.hunters_pos}
            print(q)
            print(f"{len(self.hunters)} / {len(state.my_ships)}")

        return

    def set_ship_targets(self, state):
        # we choose new targets from a pool of opponent ships. to select the
        # new targets we consider a score composed of "vulnerability" and
        # cargo. to measure vulnerability, we construct a graph where the
        # positions of our hunters have higher weights. the ships that are
        # farthest from their own shipyards are the most vulnerable
        weights = np.ones_like(state.sites)
        if self.hunters_pos.size != 0:
            weights += GRAPH_OPP_WEIGHT * \
                np.sum(state.dist[self.hunters_pos, :] <= 2, axis=0)

        graph = make_graph_csr(state, weights)

        # calculate position, halite, and vulnerability for all opponent ships
        opp_ship_pos = np.array([]).astype(int)
        opp_ship_hal = np.array([]).astype(int)
        opp_ship_vul = np.array([]).astype(int)

        for opp in state.opp_data:
            _hal, yards, ship_pos, ship_hal = opp
            ship_vul = dijkstra(graph, indices=yards, min_only=True)
            ship_vul = ship_vul[ship_pos]
            opp_ship_pos = np.append(opp_ship_pos, ship_pos)
            opp_ship_hal = np.append(opp_ship_hal, ship_hal)
            opp_ship_vul = np.append(opp_ship_vul, ship_vul)

        # if a ship has vulnerability <= 1, it will escape by reaching a yard
        # on this turn or the next, so we remove them from both from the
        # targeting pool and the targets already set
        opp_ship_pos = opp_ship_pos[opp_ship_vul > 1]
        opp_ship_hal = opp_ship_hal[opp_ship_vul > 1]
        opp_ship_vul = opp_ship_vul[opp_ship_vul > 1]

        # store positions of previous targets that are still alive
        prev = np.array([state.opp_ships[key][0] for key in self.ship_targets
                         if key in state.opp_ships]).astype(int)

        # and get the indices of the ships that are already targeted
        # the remaining ships form the pool of possible new targets
        # (note: ~ = not but numpy doesn't like not...)
        boolean_inds = np.in1d(opp_ship_pos, prev)
        target_inds = np.flatnonzero(boolean_inds)
        pool_inds = np.flatnonzero(~boolean_inds)

        # determine how many targets we would like to have
        num_ship_targets = round(len(self.hunters) / HUNTERS_PER_TARGET)

        # and add targets from the pool until the pool is empty
        # or we have enough targets. we add the untargeted ship with the
        # highest score defined as cargo * vulnerability
        opp_ship_score = opp_ship_hal * opp_ship_vul
        opp_ship_score[target_inds] = -1

        # this loop works because we set opp_ship_score < 0 at all selected
        # (target) indices so the argmax always selects a pool_ind where
        # opp_ship_score >= 0
        while target_inds.size < num_ship_targets and pool_inds.size > 0:
            new_ind = np.argmax(opp_ship_score)
            pool_inds = np.setdiff1d(pool_inds, new_ind)
            opp_ship_score[new_ind] = -1

            # only actually set the bounty if there are also some hunters
            # in the area to attack the ship
            near = state.dist[self.hunters_pos, opp_ship_pos[new_ind]] <= 3
            if np.count_nonzero(near) >= 0:
                target_inds = np.append(target_inds, new_ind)
                self.total_bounties += 1  # remove

        # stats - remove eventually
        gotem = np.array([stats.last_state.opp_ships[key][1] for key in
                          self.ship_targets if key not in state.opp_ships])
        self.conversions += gotem.size
        self.total_loot += np.sum(gotem)

        # write the new targets in the ship_targets list
        # and set position/halite/rewards for the targets
        self.ship_targets = [key for key, val in state.opp_ships.items()
                             if val[0] in opp_ship_pos[target_inds]]
        self.ship_targets_pos = opp_ship_pos[target_inds]
        self.ship_targets_hal = opp_ship_hal[target_inds]
        self.ship_targets_rew = 1000 * np.ones_like(self.ship_targets_pos)

        return

    def set_yard_targets(self, state):
        return

    def get_arrays(self, actor):
        positions = np.array([]).astype(int)
        rewards = np.array([]).astype(int)

        if actor in self.hunters:
            pos, hal = self.hunters[actor]
            inds = np.flatnonzero(self.ship_targets_hal > hal)
            positions = np.append(positions, self.ship_targets_pos[inds])
            rewards = np.append(rewards, self.ship_targets_rew[inds])

        return positions, rewards
