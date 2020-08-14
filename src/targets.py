class Targets:

    def __init__(self):
        self.nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]

        self.ship_list = []
        self.num_ships = 0

        self.distances = {}
        self.destinations = {}
        self.values = {}
        self.rankings = {}

        return

    def rates(self, state, ship):
        pos, hal = state.my_ships[ship]

        SR = BASELINE_SHIP_RATE
        YR = BASELINE_YARD_RATE

        # add a premium if there are a lot of ships that can attack us
        inds = state.opp_ship_hal < hal
        inds = inds & (state.dist[state.opp_ship_pos, pos] <= RISK_RADIUS)

        # FIX UP MORE
        YR += RISK_PREMIUM * np.sum(inds) * (state.step > STEPS_INITIAL)
        SR += RISK_PREMIUM * np.sum(inds) * (state.step > STEPS_INITIAL)

        # add a premium if we need to spawn but don't have halite
        spawn = state.my_halite < state.config.spawnCost
        spawn = spawn and should_spawn(state)
        spawn = spawn and state.step > SPAWN_PREMIUM_STEP
        SR += SPAWN_PREMIUM * spawn
        YR += SPAWN_PREMIUM * spawn

        # make the rate huge at the end of the game (ships should come home)
        if state.total_steps - state.step < STEPS_SPIKE:
            YR += SPIKE_PREMIUM
            SR += SPIKE_PREMIUM

        # hunting is risky and hunters should be very close so consider
        # both yard and ship rates
        HR = SR + YR

        # make sure all rates are < 1 so formulas remain stable
        SR = min(SR, 0.9)
        YR = min(YR, 0.9)
        HR = min(HR, 0.9)

        return SR, YR, HR

    def update(self, state, queue):
        # if there are no ships, there is nothing to do
        if len(queue.ships) == 0:
            return

        # list of ships -  indices of ships in matrices and arrays
        # will be the index in this list
        self.ship_list = list(queue.ships.keys())
        self.num_ships = len(self.ship_list)

        # calculate distances on the relevant weighted graphs
        self.distances = {ship: self.calc_distances(ship, state)
                          for ship in self.ship_list}

        # the optimal assignment will assign only one ship to each site
        # but we want more than one ship to go back to each yard so
        # we add duplicates of the yards to the rewards to ensure this
        self.duplicates = np.tile(state.my_yard_pos, self.num_ships - 1)
        self.ind_to_site = np.append(self.duplicates, state.sites)

        # calculate the value per step of going to a specific site
        cost_matrix = np.vstack([self.calc_rewards(ship, state)
                                 for ship in self.ship_list])

        # find the optimal assignment of ships to destinations
        # the optimal assignment assignment assigns ship_inds[i] to
        # site_inds[i]
        ship_inds, site_inds = assignment(cost_matrix, maximize=True)

        # go through the solution of the optimal assignment problem and
        # extract destinations, values, and move ranking functions
        # for each ship
        self.destinations = {}
        self.values = {}
        self.rankings = {}

        for ship_ind, site_ind in zip(ship_inds, site_inds):
            # convert indices into ships, sites, and values and
            # add the destination
            site = self.ind_to_site[site_ind]
            ship = self.ship_list[ship_ind]
            value = cost_matrix[ship_ind, site_ind]
            self.add_destination(ship, site, value)

        return

    def calc_rewards(self, ship, state, return_appended=True):
        reward_map = np.zeros_like(state.halite_map)

        hood_dists, yard_dists = self.distances[ship]
        ship_dists = hood_dists[self.nnsew.index(None), :]

        # determine ship rate, yard rate, and hunting rate
        SR, YR, HR = self.rates(state, ship)

        # add rewards for mining at all minable sites
        # see notes for explanation of these quantities
        C = state.my_ships[ship][1]

        # indices of minable sites
        minable = state.halite_map > MIN_MINING_HALITE

        # # ignore halite next to enemy yards so ships don't get poached
        if state.opp_yard_pos.size != 0:
            no_yards = np.amin(state.dist[state.opp_yard_pos, :], axis=0) > 1
            minable = minable & no_yards

        H = state.halite_map[minable]
        SD = ship_dists[minable]
        YD = yard_dists[minable]

        X = (1 + state.config.regenRate) * (1 - state.config.collectRate)
        A = state.config.collectRate / (1 - X)

        F = ((1 + SR) ** SD) * ((1 + YR) ** YD)
        F1 = C / F
        F2 = A * ((1 + state.config.regenRate) ** SD) * H
        F2 = F2 / F

        with np.errstate(divide='ignore'):
            M = np.log(1 + F1 / F2) - np.log(1 - np.log(X) / np.log(1 + YR))

        M = np.fmax(1, np.round(M / np.log(X)))

        reward_map[minable] = (F1 + F2 * (1 - X ** M)) / ((1 + YR) ** M)

        # add rewards at yard for depositing halite
        ship_yard_dist = ship_dists[state.my_yard_pos]
        reward_map[state.my_yard_pos] = C / ((1 + YR) ** ship_yard_dist)

        # insert bounties for opponent ships
        ship_hunt_pos, ship_hunt_rew = bounties.get_ship_targets(ship, state)
        discount = (1 + HR) ** ship_dists[ship_hunt_pos]
        reward_map[ship_hunt_pos] = ship_hunt_rew / discount

        # insert bounties for opponent yards
        yard_hunt_pos, yard_hunt_rew = bounties.get_yard_targets(ship, state)
        discount = (1 + HR) ** ship_dists[yard_hunt_pos]
        reward_map[yard_hunt_pos] = yard_hunt_rew / discount

        if return_appended:
            # copy the ship yard rewards onto the duplicate ship yards and
            # append the duplicate rewards
            yard_rewards = reward_map[state.my_yard_pos]
            duplicate_rewards = np.tile(yard_rewards, self.num_ships - 1)
            return np.append(duplicate_rewards, reward_map)
        else:
            return reward_map

    def add_destination(self, ship, site, value):
        # store site and value
        self.destinations[ship] = site
        self.values[ship] = value

        # then store a copy of nnsew, sorted by how much the move decreases
        # the distance to our target. give a 1/2 penalty to None, so that we
        # move instead of staying put if there is no difference in the distance
        hood_dists = self.distances[ship][0]
        dest_dists = hood_dists[:, site]
        dist_after = lambda move: dest_dists[self.nnsew.index(move)] \
                   + (move is None) / 2
        self.rankings[ship] = self.nnsew.copy()
        self.rankings[ship].sort(key=dist_after)

        return

    def calc_distances(self, ship, state):
        pos = state.my_ships[ship][0]

        # construct a weighted graph to calculate distances on
        weights = self.make_weights(ship, state)
        graph = self.make_graph_csr(state, weights)

        # calculate the distance from all sites to the ship
        # also calculate the distance from all sites to the immediate
        # neighbors of ship - then we can easily take a step "towards"
        # any given site later
        hood = np.array([state.newpos(pos, move) for move in self.nnsew])
        hood_dists = dijkstra(graph, indices=hood)

        # calculate the distances from all sites to nearest yard
        yard_dists = dijkstra(graph, indices=state.my_yard_pos, min_only=True)

        # store hood distances and yard distances for later
        return hood_dists, yard_dists

    def make_weights(self, actor, state):
        pos, hal = state.my_ships[actor]
        weights = np.ones_like(state.sites)

        # ignore immediate neighbors in the friendly weights - this is handled
        # automatically and the weights can cause traffic jams at close range
        # also ignore weights of any ships on fifo yards
        friendly = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos)
        friendly = friendly[state.dist[pos, friendly] > 1]
        if friendly.size != 0:
            hood = state.dist[friendly, :] <= MY_RADIUS
            weights += MY_WEIGHT * np.sum(hood, axis=0)

        # only consider opponent ships with less halite
        threats = state.opp_ship_pos[state.opp_ship_hal <= hal]
        if threats.size != 0:
            hood = state.dist[threats, :] <= OPP_RADIUS
            weights += OPP_WEIGHT * np.sum(hood, axis=0)

        # also need to go around enemy shipyards
        if state.opp_yard_pos.size != 0:
            weights[state.opp_yard_pos] += OPP_WEIGHT

        # remove any weights on the yards so these don't get blocked
        weights[state.my_yard_pos] = 0

        return weights

    def make_graph_csr(self, state, weights):
        nsites = state.map_size ** 2

        # weight at any edge (x,y) is (w[x] + w[y])/2
        site_weights = weights[state.sites]
        n_weights = 0.5 * (site_weights + weights[state.north])
        s_weights = 0.5 * (site_weights + weights[state.south])
        e_weights = 0.5 * (site_weights + weights[state.east])
        w_weights = 0.5 * (site_weights + weights[state.west])

        # column indices for row i are in indices[indptr[i]:indptr[i+1]]
        # and their corresponding values are in data[indptr[i]:indptr[i+1]]
        indptr = 4 * state.sites
        indptr = np.append(indptr, 4 * nsites)
        indices = np.vstack((state.north, state.south, state.east, state.west))
        indices = np.ravel(indices, "F")
        data = np.vstack((n_weights, s_weights, e_weights, w_weights))
        data = np.ravel(data, "F")

        return csr_matrix((data, indices, indptr), shape=(nsites, nsites))
