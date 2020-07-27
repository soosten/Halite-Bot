class Targets:

    def __init__(self):
        self.nnsew = [None, "NORTH", "SOUTH", "EAST", "WEST"]

        self.ship_list = []
        self.distances = {}
        self.rewards = {}
        self.destinations = {}
        self.values = {}
        self.rankings = {}

        return

    def calculate(self, state, queue):
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
        self.rewards = {ship: self.calc_rewards(ship, state)
                        for ship in self.ship_list}

        # find the optimal assignment of ships to destinations
        # the optimal assignment assignment assigns ship_inds[i] to
        # site_inds[i]
        problem = np.vstack([self.rewards[ship] for ship in self.ship_list])
        ship_inds, site_inds = assignment(problem, maximize=True)

        # go through the solution of the optimal assignment problem and
        # extract destinations, values, and move ranking functions
        # for each ship
        self.destinations = {}
        self.values = {}
        self.rankings = {}

        for ship_ind, site_ind in zip(ship_inds, site_inds):
            # first write destination and values
            ship = self.ship_list[ship_ind]
            self.destinations[ship] = self.ind_to_site[site_ind]
            self.values[ship] = self.rewards[ship][site_ind]

            # then store a copy of nnsew, sorted by how much the move
            # decreases the distance to our target
            hood_dists = self.distances[ship][0]
            dest_dists = hood_dists[:, self.destinations[ship]]
            dist_after = lambda x: dest_dists[self.nnsew.index(x)]
            self.rankings[ship] = self.nnsew.copy()
            self.rankings[ship].sort(key=dist_after)

        return

    def calc_rewards(self, ship, state, unappended=False):
        hal = state.my_ships[ship][1]

        # get the relevant distances
        hood_dists, yard_dists = self.distances[ship]
        ship_dists = hood_dists[self.nnsew.index(None), :]
        D = ship_dists + yard_dists

        # calculate halite gain per step that results from going
        # to a site, mining it, and returning to the nearest yard,
        # optimized over the numer of steps to mine the site for
        # see the notes for a detailed explanation
        x = (1 + state.config.regenRate) * (1 - state.config.collectRate)
        a = 1 / (1 - x)
        r = self.interest(ship, state)

        # the sites for which the mining optimum makes sense
        minable = state.halite_map > 0
        H = state.halite_map[minable]

        # calculate the optimal turn to mine each site M
        alpha = 1 + hal / (a * H)
        xM = alpha / (1 - (np.log(x) / np.log(1 + r)))
        M = np.ones_like(state.halite_map)
        M[minable] = np.log(xM) / np.log(x)
        M[minable] = np.fmax(np.round(M[minable]), 1)

        # set M = 0 on yards so that there is a preference for the yard
        # over the site next to the yard in D+M
        M[state.my_yard_pos] = 0

        # plug the optimizing Ms into the formula for halite
        # returned per step after traveling to each site
        reward_map = hal + a * (1 - x ** M) * state.halite_map

        # insert the reward of going home to a shipyard
        reward_map[state.my_yard_pos] = hal

        # insert bounties for opponent ships
        ship_hunt_pos, ship_hunt_rew = bounties.get_ship_targets(ship, state)
        reward_map[ship_hunt_pos] += ship_hunt_rew

        # insert bounties for opponent yards
        yard_hunt_pos, yard_hunt_rew = bounties.get_yard_targets(ship, state)
        reward_map[yard_hunt_pos] += yard_hunt_rew

        # discount each reward by how much time is lost by retrieving it
        discounts = (1 + r) ** (D + M)
        reward_map = reward_map * (1 + r) / discounts

        # copy the ship yard rewards onto the duplicate ship yards
        yard_rewards = reward_map[state.my_yard_pos]
        duplicate_rewards = np.tile(yard_rewards, self.num_ships - 1)

        # append the duplicate rewards
        appended = np.append(duplicate_rewards, reward_map)

        if unappended:
            return appended, reward_map
        else:
            return appended

    def interest(self, ship, state):
        # always have a minimum interest rate
        rate = BASELINE_INTEREST

        # add a premium if there are a lot of ships that can attack us
        hal = state.my_ships[ship][1]
        less_hal = state.opp_ship_pos[state.opp_ship_hal < hal]
        rate += RISK_PREMIUM * less_hal.size

        # add a premium of we want to spawn and need money
        if should_spawn(state) and state.my_halite < state.config.spawnCost:
            rate += SPAWN_PREMIUM

        # make the rate huge at the end of the game (ships should come home)
        if state.total_steps - state.step < STEPS_SPIKE:
            rate += 0.7

        # make sure the rate is no too close to 1 so formulas stay stable
        rate = min(rate, 0.9)

        return rate

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
        hood_dists = dijkstra(graph, indices=hood, directed=False)

        # calculate the distances from all sites to nearest yard
        yard_dists = dijkstra(graph, indices=state.my_yard_pos,
                              min_only=True, directed=False)

        # store hood distances and yard distances for later
        return hood_dists, yard_dists

    def make_weights(self, actor, state):
        weights = np.ones_like(state.sites)

        # ships contribute weights in the space they control (which is the
        # ball of radius 1 or 2 around their position). sites controlled by
        # multiple ships should get higher weights

        # heuristic: going "through a site" usually takes two steps. if you
        # to go "around the site" while staying 1 step away it takes 4 steps
        # so the weight should be > 4/2 = 2
        mw = GRAPH_MY_WEIGHT

        # going "through 3 sites" usually takes 4 steps. if you want to go
        # want "around the 3 sites" while staying 2 steps from the middle, it
        # takes 8 steps so the weight should be > 8/4 = 2. but we want to be
        # very scared of opponent ships so we set this to 4
        ow = GRAPH_OPP_WEIGHT

        pos, hal = state.my_ships[actor]

        # ignore immediate neighbors in the friendly weights - this is handled
        # automatically and the weights can cause traffic jams at close range
        # also ignore weights of any ships on fifo yards
        friendly = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos)
        friendly = friendly[state.dist[pos, friendly] > 1]
        if friendly.size != 0:
            weights += mw * np.sum(state.dist[friendly, :] <= 1, axis=0)

        # only consider opponent ships with less halite
        less_hal = state.opp_ship_pos[state.opp_ship_hal <= hal]
        if less_hal.size != 0:
            weights += ow * np.sum(state.dist[less_hal, :] <= 2, axis=0)

        # also need to go around enemy shipyards
        if state.opp_yard_pos.size != 0:
            weights[state.opp_yard_pos] += ow

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
