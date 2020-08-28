class Targets:
    def __init__(self, state, actions, bounties, spawns):
        self.num_ships = len(actions.ships)

        # if there are no ships, there is nothing to do
        if self.num_ships == 0:
            return

        # read relevant spawning information
        self.spawns_wanted = spawns.ships_wanted
        self.spawns_possible = spawns.ships_possible
        likely_spawns = spawns.spawn_pos[0:self.spawns_possible]
        self.protected = np.setdiff1d(memory.protected, likely_spawns)

        # set up candidate moves for each ship and compute
        # distances on an appropriately weighted graph
        self.geometry(state, actions)

        # the optimal assignment will assign only one ship to each site
        # but we want more than one ship to go back to each yard so
        # we add duplicates of the yards to the rewards to make this possible
        duplicates = np.tile(state.my_yard_pos, self.num_ships - 1)
        ind_to_site = np.append(duplicates, state.sites)

        # calculate the value of going to a site for each ship
        cost_matrix = np.vstack([self.rewards(ship, state, bounties)
                                 for ship in actions.ships])

        # find the optimal assignment of ships to destinations
        # the optimal assignment assigns ship_inds[i] to site_inds[i]
        ship_inds, site_inds = assignment(cost_matrix, maximize=True)

        # go through the solution of the optimal assignment problem and
        # order the moves by preference
        self.destinations = {}
        self.values = {}

        for ship_ind, site_ind in zip(ship_inds, site_inds):
            # store destination and value of the ship
            ship = actions.ships[ship_ind]
            self.destinations[ship] = ind_to_site[site_ind]
            self.values[ship] = cost_matrix[ship_ind, site_ind]

            # sort moves by how much they decrease the distance
            # to the assigned destination
            dest_dists = self.move_dists[ship][:, self.destinations[ship]]
            self.moves[ship] = self.moves[ship][dest_dists.argsort()]

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
        spawn = (self.spawns_wanted > 0) and (self.spawns_possible == 0)
        spawn = spawn and (state.step > SPAWN_PREMIUM_STEP)
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

    def rewards(self, ship, state, bounties):
        pos, hal = state.my_ships[ship]

        reward_map = np.zeros_like(state.halite_map)

        pos_ind = np.flatnonzero(self.moves[ship] == pos)[0]
        ship_dists = self.move_dists[ship]
        ship_dists = ship_dists[pos_ind, :]

        # determine ship rate, yard rate, and hunting rate
        SR, YR, HR = self.rates(state, ship)

        # add rewards for mining at all minable sites
        # see notes for explanation of these quantities
        C = state.my_ships[ship][1]

        # indices of minable sites
        minable = state.halite_map > MIN_MINING_HALITE

        # ignore halite next to enemy yards so ships don't get poached
        opp_yard_dist = np.amin(state.dist[state.opp_yard_pos, :], axis=0,
                                initial=state.map_size)
        minable &= (opp_yard_dist > 1)

        H = state.halite_map[minable]
        SD = ship_dists[minable]
        YD = self.yard_dists[ship][minable]

        X = (1 + state.regen_rate) * (1 - state.collect_rate)
        A = state.collect_rate / (1 - X)

        F = ((1 + SR) ** SD) * ((1 + YR) ** YD)
        F1 = C / F
        F2 = A * ((1 + state.regen_rate) ** SD) * H
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

        # copy the ship yard rewards onto the duplicate ship yards and
        # append the duplicate rewards
        yard_rewards = reward_map[state.my_yard_pos]
        duplicate_rewards = np.tile(yard_rewards, self.num_ships - 1)

        # finally put a large bonus on going to any protected yards that
        # the ships is next to - this ensures one close ship always
        # chooses the yard - but don't copy this bonus on the duplicates
        # since only one ship needs to be there
        inds = state.dist[self.protected, pos] <= 1
        reward_map[self.protected[inds]] += 1000

        return np.append(duplicate_rewards, reward_map)

    def geometry(self, state, actions):
        self.moves = {}
        self.move_dists = {}
        self.yard_dists = {}

        for ship in actions.ships:
            pos, hal = state.my_ships[ship]
            self.moves[ship] = np.flatnonzero(state.dist[pos, :] <= 1)

            # construct a weighted graph to calculate distances on
            weights = self.make_weights(ship, state)
            graph = self.make_graph_csr(state, weights)

            # distance to each site after making a move
            self.move_dists[ship] = dijkstra(graph, indices=self.moves[ship])

            # calculate the distances from all sites to nearest yard
            # if there are no yards, take the maximum cargo ship instead
            # which is the ship most likely to convert soon
            if state.my_yard_pos.size != 0:
                yard_pos = state.my_yard_pos
            else:
                yard_pos = state.my_ship_pos[state.my_ship_hal.argmax()]

            self.yard_dists[ship] = dijkstra(graph, indices=yard_pos,
                                             min_only=True)

        return

    def make_weights(self, actor, state):
        pos, hal = state.my_ships[actor]
        weights = np.ones_like(state.sites)

        # ignore immediate neighbors in the friendly weights - this is handled
        # automatically and the weights can cause traffic jams at close range
        # also ignore weights of any ships on fifo yards
        # friendly = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos)
        friendly = state.my_ship_pos
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
        weights[state.opp_yard_pos] += OPP_WEIGHT

        # remove any weights on the yards so these don't get blocked
        weights[state.my_yard_pos] = 0

        return weights

    def make_graph_csr(self, state, weights):
        # weight at any edge (x,y) is (w[x] + w[y])/2

        nsites = state.map_size ** 2

        # column indices for row i are in indices[indptr[i]:indptr[i+1]]
        # and their corresponding values are in data[indptr[i]:indptr[i+1]]
        indptr = 4 * np.append(state.sites, nsites)

        indices = np.empty(4 * nsites, dtype=int)
        indices[0::4] = state.north
        indices[1::4] = state.south
        indices[2::4] = state.east
        indices[3::4] = state.west

        data = np.empty(4 * nsites, dtype=float)
        data[0::4] = 0.5 * (weights + weights[state.north])
        data[1::4] = 0.5 * (weights + weights[state.south])
        data[2::4] = 0.5 * (weights + weights[state.east])
        data[3::4] = 0.5 * (weights + weights[state.west])

        return csr_matrix((data, indices, indptr), shape=(nsites, nsites))
