class Targets:
    def __init__(self):
        self.targets = []
        self.hunters = []
        self.targets_pos = np.array([]).astype(int)
        self.targets_hal = np.array([]).astype(int)
        self.rewards = np.array([]).astype(int)
        self.conversions = 0
        self.total_bounties = 0
        self.total_loot = 0
        self.internal = None
        return

    def get_arrays(self, actor):
        # check if our ship is a designated hunter and if any bounties have
        # been set. if so, add those targets with higher cargo than ours
        # to the map of rewards
        if actor in self.hunters and self.not_empty():
            pos, hal = self.internal.my_ships[actor]
            inds = np.flatnonzero(self.targets_hal > hal)
            if inds.size != 0:
                return self.targets_pos[inds], self.rewards[inds]
            else:
                return np.array([]), np.array([])

        return np.array([]), np.array([])

    def not_empty(self):
        return len(self.targets) > 0

    def update(self, state):
        self.internal = deepcopy(state)
        # decide which of our ships should hunt depending on their cargo
        # and how much halite is on the map
        if state.my_ship_pos.size != 0:
            q = MAX_HUNTING_QUANTILE \
                - np.sum(state.halite_map) / state.config.startingHalite
            q = min(1, max(0, q))  # quantiles should be in [0,1]
            cutoff = np.quantile(state.my_ship_hal, q)
            self.hunters = [ship for ship in state.my_ships.keys()
                            if state.my_ships[ship][1] < cutoff
                            and state.my_ships[ship][0] not in fifos.fifo_pos]
        else:
            self.hunters = []

        # determine how many targets we should have
        n_targets = len(self.hunters) // HUNTERS_PER_TARGET

        # remove any targets which are no longer eligible
        self.targets = [target for target in self.targets
                        if self.eligible(target, state)]

        # create a pool of untargeted opponent ships
        pool = [target for target in state.opp_ships if
                self.eligible(target, state) and target not in self.targets]

        # add new targets with highest values until we have n_targets targets
        while len(self.targets) < n_targets and len(pool) > 0:
            new_target = max(pool, key=lambda x: self.attack_value(x, state))
            pool.remove(new_target)
            if self.attack_value(new_target, state) > 0:
                self.total_bounties += 1
                self.targets.append(new_target)

        # update the target positions, cargos, and rewards
        self.targets_pos = np.array([state.opp_ships[ship][0]
                                     for ship in self.targets]).astype(int)
        self.targets_hal = np.array([state.opp_ships[ship][1]
                                     for ship in self.targets]).astype(int)
        self.rewards = 1000 * np.ones_like(self.targets_pos)
        return

    def eligible(self, target, state):
        # check if the ship is still alive
        if target not in state.opp_ships.keys():
            self.conversions += 1
            self.total_loot += stats.last_state.opp_ships[target][1]
            return False

        # if it got away by getting too close to its own yards to pursue
        if self.dist_to_yard(target, state) <= 1:
            return False

        # if it got away by outrunning its pursuers
        if self.attack_value(target, state) <= 0:  # RETHINK?
            return False

        return True

    def attack_value(self, target, state):
        pos, hal = state.opp_ships[target]

        hunter_pos = np.array([state.my_ships[ship][0] for ship in
                               self.hunters if state.my_ships[ship][1] < hal])
        if hunter_pos.size != 0:
            close_hunters = np.sum(state.dist[hunter_pos, pos] <= 3)
        else:
            close_hunters = 0

        if close_hunters < 3:  # or self.dist_to_yard(target, state) < 3:
            return -1
        else:
            return close_hunters * hal

    def dist_to_yard(self, target, state):
        # find the yards corresponding to the target ship
        # (state.opp_data is a list of lists...)
        pos = state.opp_ships[target][0]
        opp = [data for data in state.opp_data if pos in data[2]]
        yards = opp[0][1]

        # return the distance of the yards to the position of the ship
        if yards.size == 0:
            return np.inf
        else:
            return np.min(state.dist[yards, pos])
