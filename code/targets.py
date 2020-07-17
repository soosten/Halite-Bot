class Targets:
    def __init__(self):
        self.hunters = {}
        self.hunters_pos = np.array([]).astype(int)
        self.hunters_hal = np.array([]).astype(int)

        self.ship_targets = []
        self.ship_targets_pos = np.array([]).astype(int)
        self.ship_targets_hal = np.array([]).astype(int)
        self.ship_targets_rew = np.array([]).astype(int)

        # stats - remove
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
            # the cells around x.
            boolhood = state.dist[state.my_ship_pos, :] <= 3
            hood = np.apply_along_axis(np.flatnonzero, 1, boolhood)
            hood_hal = np.mean(state.halite_map[hood], axis=1)
            ship_val = state.my_ship_hal + hood_hal

            # choose a fraction of ships depending on halite on the map
            q = MAX_HUNTERS_PER_SHIP
            q -= np.sum(state.halite_map) / state.config.startingHalite
            q = q if q > MIN_HUNTERS_PER_SHIP else 0
            num_hunters = int(q * len(state.my_ships))

            # designate the num_hunters ships with lowest value as hunters
            # and remove any fifo ships from the hunters
            # if num_hunters = 0, we pass -1 to argpartition which sorts from
            # the other end - but we don't care since the slice 0:num_hunters
            # will be empty...
            hunters_inds = np.argpartition(ship_val, num_hunters - 1)
            hunters_inds = hunters_inds[0:num_hunters]
            fifo_bool = np.in1d(state.my_ship_pos, fifos.fifo_pos)
            fifo_inds = np.flatnonzero(fifo_bool)
            hunters_inds = np.setdiff1d(hunters_inds, fifo_inds)
            self.hunters_pos = state.my_ship_pos[hunters_inds]
            self.hunters_hal = state.my_ship_hal[hunters_inds]

            # also keep hunters in dictionary form like state.my_ships
            self.hunters = {key: val for key, val in state.my_ships.items()
                            if val[0] in self.hunters_pos}

        return

    def set_ship_targets(self, state):
        # we choose new targets from a pool of opponent ships. to select the
        # new targets we consider a score composed of "vulnerability" and
        # cargo. to measure vulnerability, we construct a graph where the
        # positions of our hunters have higher weights. the ships that are
        # farthest from their own shipyards are the most vulnerable
        weights = np.ones_like(state.sites)
        if self.hunters_pos.size != 0:
            hood = state.dist[self.hunters_pos, :] <= 3
            weights += GRAPH_MY_WEIGHT * np.sum(hood, axis=0)

        graph = make_graph_csr(state, weights)

        # calculate position, halite, and vulnerability for all opponent ships
        opp_ship_pos = np.array([]).astype(int)
        opp_ship_hal = np.array([]).astype(int)
        opp_ship_vul = np.array([]).astype(int)

        for opp in state.opp_data:
            _hal, yards, ship_pos, ship_hal = opp
            # if yards is empty, dijkstra returns inf. cut this off so
            # that 0 * opp_ship_vul = 0 (instead of nan)
            ship_vul = dijkstra(graph, indices=yards, min_only=True)
            ship_vul = np.fmin(ship_vul[ship_pos], 10000).astype(int)
            opp_ship_pos = np.append(opp_ship_pos, ship_pos)
            opp_ship_hal = np.append(opp_ship_hal, ship_hal)
            opp_ship_vul = np.append(opp_ship_vul, ship_vul)

        nearby = state.dist[np.ix_(self.hunters_pos, opp_ship_pos)] <= 3
        less_hal = self.hunters_hal[:, np.newaxis] < opp_ship_hal
        nearby = np.sum(nearby & less_hal, axis=0)

        # store current positions of previous targets that are still alive
        prev = np.array([val[0] for key, val in state.opp_ships.items()
                         if key in self.ship_targets]).astype(int)

        # get the indices of the ships that are already targeted
        # if a ship has vulnerability <= 4, it will most likely escape by
        # reaching a yard soon, so we remove such ships from the targets
        # (note: & / | / ~ = and / or / not in numpy compatible way)
        target_bool = np.in1d(opp_ship_pos, prev) & (opp_ship_vul >= 4)
        target_inds = np.flatnonzero(target_bool)

        # the pool of possible new targets consists of non-targeted ships that
        # also satisfy some further conditions
        candidates = ~target_bool
        candidates = candidates & (opp_ship_vul >= 4) & (nearby >= 3)

        # we compute scores for each of the candidate ships indicating
        # the risk/reward of attacking them
        # make the scores of ships that are not candidates negative
        opp_ship_score = opp_ship_hal * opp_ship_vul
        opp_ship_score[~candidates] = -1

        # determine how many targets we would like to have and how many
        # new targets we should/can build
        num_targets = len(self.hunters) // HUNTERS_PER_TARGET
        num_new_targets = max(num_targets - target_inds.size, 0)
        num_new_targets = min(num_new_targets, np.sum(candidates))

        # we can take those num_new_targets ships with maximum score
        # since scores are >= 0  and we forced the scores of non-candidate
        # ships to equal -1. see comment before argpartition in set_hunters
        new_inds = np.argpartition(-opp_ship_score, num_new_targets - 1)
        target_inds = np.append(target_inds, new_inds[0:num_new_targets])

        # stats - remove eventually
        self.total_bounties += num_new_targets
        gotem = np.array([stats.last_state.opp_ships[key][1] for key in
                          self.ship_targets if key not in state.opp_ships])
        self.conversions += gotem.size
        self.total_loot += np.sum(gotem)

        # set position/halite/rewards for the targets
        self.ship_targets_pos = opp_ship_pos[target_inds]
        self.ship_targets_hal = opp_ship_hal[target_inds]
        self.ship_targets_rew = 1000 * np.ones_like(self.ship_targets_pos)

        # write the new targets in the ship_targets list
        self.ship_targets = [key for key, val in state.opp_ships.items()
                             if val[0] in self.ship_targets_pos]

        return

    def set_yard_targets(self, state):
        return

    def get_arrays(self, actor, state):
        # append targets/rewards here depending on the nature of the ship
        positions = np.array([]).astype(int)
        rewards = np.array([]).astype(int)

        # if the ship is supposed to hunt, add all targets within reach
        # that also have more cargo than we do
        if actor in self.hunters:
            pos, hal = self.hunters[actor]
            hal_inds = self.ship_targets_hal > hal
            pos_inds = state.dist[self.ship_targets_pos, pos] <= 5
            inds = np.flatnonzero(pos_inds & hal_inds)
            positions = np.append(positions, self.ship_targets_pos[inds])
            rewards = np.append(rewards, self.ship_targets_rew[inds])

        # if the ship is supposed to target shipyards, add these

        return positions, rewards
