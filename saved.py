def opp_ship_collision(self, ship, action, strict=True):
    # return True if a collision only with opponent SHIPS (not yards) is
    # possible in the next step if ship performs action. this is for
    # attacking shipyards

    collision = False

    pos, hal = self.my_ships[ship]
    npos = self.newpos(pos, action)

    if strict:
        less_hal = self.opp_ship_pos[self.opp_ship_hal <= hal]
    else:
        less_hal = self.opp_ship_pos[self.opp_ship_hal < hal]

    # less_hal could be empty
    if less_hal.size != 0:
        # list of sites where ships in less_hal could end up after one turn
        hood = np.flatnonzero(np.amin(self.dist[less_hal, :], axis=0) <= 1)

        # regard our own shipyards as safe havens - this is not necessarily
        # the case but many opponents will not want to run into the yard
        # this will cost them their ship too
        hood = np.setdiff1d(hood, self.my_yard_pos)

        collision = npos in hood

    return collision



# maximum number of ships per turn which attack enemy shipyards
MAX_SHIPYARD_ATTACKERS = 0

    # check whether this ship should attack yards - if so, we set appropriate
    # targets and update the map of rewards
    targets = attack_shipyards(actor, state)
    if targets.size != 0:
        rewards[targets] = state.config.convertCost


def attack_shipyards(actor, state):
    # don't attack shipyards unless configuration allows it
    if MAX_SHIPYARD_ATTACKERS == 0:
        return np.array([])

    # don't attack shipyards when we need to build ships - this tactic
    # is mainly to disrupt turtling strategies towards the end of the game
    if should_spawn(state) or state.step <= 300:
        return np.array([])

    # don't attack shipyards when there is a lot of halite to be mined
    hal_frac = np.sum(state.halite_map) / state.config.startingHalite
    if hal_frac > 1/10:
        return np.array([])

    # don't attack shipyards if we have cargo
    pos, hal = state.my_ships[actor]
    if hal != 0:
        return np.array([])

    # otherwise find the shipyards of the opponent whose score is closest to
    # ours (score = stored halite + 2/3 * cargo)
    my_score = state.my_halite + (2/3) * np.sum(state.my_ship_hal)
    score_diff = lambda x: np.abs(x[0] + (2/3) * np.sum(x[3]) - my_score)
    competitor = max(state.opp_data, key=score_diff)
    targets = competitor[1]

    # if we already got all the shipyards, don't attack anything else
    if targets.size == 0:
        return np.array([])

    # find our ships with no cargo that are closest to the targets
    # we know that there are ships with no cargo, since actor is one of them
    # if execution gets here
    no_hal_pos = state.my_ship_pos[state.my_ship_hal == 0]
    target_dists = np.amin(state.dist[np.ix_(targets, no_hal_pos)], axis=0)

    num_attackers = min(MAX_SHIPYARD_ATTACKERS, no_hal_pos.size)
    partition = np.argpartition(target_dists, num_attackers - 1)
    attackers = no_hal_pos[partition[0:num_attackers]]

    return targets if pos in attackers else np.array([])


HUNT_MIN_SAMPLES = 3
HUNT_MIN_CLUSTER_SIZE = 5

def attack_ships(actor, state):
    pos, hal = state.my_ships[actor]

    # if the ship is not a hunter, don't set any targets
    if pos not in state.hunters:
        return np.array([]), np.array([])

    # otherwise get the cluster of hunters with a common target
    cluster = state.get_hunting_cluster(pos)
    ship_indices = np.flatnonzero(np.in1d(state.my_ship_pos, cluster))
    clust_indices = np.flatnonzero(np.in1d(state.hunters, cluster))

    # don't hunt solo (for now)
    if cluster.size == 1:
        return np.array([]), np.array([])

    # determine which opponent ships can be attacked by this cluster
    max_hal = np.max(state.my_ship_hal[ship_indices])
    prey = state.opp_ship_pos[state.opp_ship_hal > max_hal]

    # if there is no common prey, don't set targets
    if prey.size == 0:
        return np.array([]), np.array([])

    # choose the target with the minimum average distance
    avg_dists = np.sum(state.hunting_dists[np.ix_(clust_indices, prey)], axis=0)
    target_pos = prey[avg_dists.argmin()]

    targets = np.array([target_pos])
    values = state.config.spawnCost * np.ones_like(targets)

    # targets = np.flatnonzero(state.dist[target_pos, :] <= 1)
    # values = state.config.spawnCost * np.ones_like(targets)

    return targets, values


  def _make_hunting_clusters(self):
        # decide how many of our ships should be hunters
        frac = max(0, 0.5 - np.sum(self.halite_map) / self.config.startingHalite)
        n_hunters = np.floor(frac * self.my_ship_pos.size).astype(int) # fix!

        # if there are no hunters, return empty clusters
        if n_hunters == 0:
            self.hunters = np.array([])
            self.hunter_labels = np.array([])
            return
        # otherwise choose the n_hunters ships with lowest cargo
        else:
            indices = np.argpartition(self.my_ship_hal, n_hunters - 1)
            indices = indices[0:n_hunters]
            self.hunters = self.my_ship_pos[indices]

        # if there are not enough sites to cluster, set them all as outliers
        if self.hunters.size <= HUNT_MIN_CLUSTER_SIZE:
            self.hunter_labels = - np.ones_like(self.hunters)
            return

        # otherwise we construct graph weights
        # we put a weight on neighborhoods of the opponent ships with less
        # than the average cargo of the hunters and on the yards
        opp_weight = GRAPH_OPP_WEIGHT

        weights = np.ones(self.map_size ** 2)

        avg_hunter_hal = np.mean(self.my_ship_hal[indices])
        obstacles = self.opp_ship_pos[self.opp_ship_hal <= avg_hunter_hal]

        if obstacles.size != 0:
            weights += opp_weight * \
                np.sum(self.dist[obstacles, :] <= 2, axis=0)

        if self.opp_yard_pos.size != 0:
            weights[self.opp_yard_pos] += opp_weight

        graph = make_graph_csr(self, weights)

        # compute graph distances to all hunters
        self.hunting_dists = dijkstra(graph, indices=self.hunters)
        dist_matrix = self.hunting_dists[:, self.hunters]

        # run the sklearn clustering algorithm
        model = OPTICS(min_samples=HUNT_MIN_SAMPLES,
                       min_cluster_size=HUNT_MIN_CLUSTER_SIZE,
                       metric="precomputed")

        # ignore outlier warnings from OPTICS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hunter_labels = model.fit_predict(dist_matrix)

        return

    def get_hunting_cluster(self, pos):
        if pos not in self.hunters:
            print("Error: hunter site not found")
            return -1

        ind = np.flatnonzero(self.hunters == pos)
        label = self.hunter_labels[ind]

        if label == -1:  # outliers - return only the position
            return np.array([pos])
        else:
            cluster_inds = np.flatnonzero(self.hunter_labels == label)
            return self.hunters[cluster_inds]

# still a hack
def should_spawn(actor, state):
    # calculate how much space we control versus opponents
    # any ship or yard controls all sites in B(x, 2)_L1 where
    # x is its position
    opp_sites = np.append(state.opp_ship_pos, state.opp_yard_pos)
    opp_sites = np.flatnonzero(np.amin(state.dist[opp_sites, :], axis=0) <= 2)

    my_sites = np.append(state.my_ship_pos, state.my_yard_pos)
    my_sites = np.flatnonzero(np.amin(state.dist[my_sites, :], axis=0) <= 2)

    free = np.setdiff1d(state.sites, np.union1d(my_sites, opp_sites))
    overlap = np.intersect1d(opp_sites, my_sites)
    opp_sites = np.setdiff1d(opp_sites, overlap)
    my_sites = np.setdiff1d(my_sites, overlap)

    scores = [x[0] for x in state.opp_data]

    if state.my_halite > 1000 * (state.step/state.total_steps) \
                       + max(scores):
        ships_wanted = (free.size + my_sites.size) / 10
    else:
        ships_wanted = (free.size + my_sites.size) / 16

    ships_wanted = max(10, ships_wanted)

    if ships_wanted < len(state.my_ships):
        return False

    # don't spawn any more ships towards the end of the game
    # they won't pay for their cost in time
    if state.total_steps - state.step < state.map_size:
        return False

    # check how many ships are right next to the yard and return false
    # if there are already 3 ships in the way
    pos = state.my_yards[actor]
    dist1 = np.count_nonzero(state.dist[state.my_ship_pos, pos] <= 1)
    if dist1 > 2:
        print("checking mattered!")
        return False

    # otherwise, we want a ship
    return True
