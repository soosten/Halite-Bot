class Bounties:
    def __init__(self):
        self.ship_targets = []
        self.ship_targets_pos = np.array([]).astype(int)
        self.ship_targets_hal = np.array([]).astype(int)
        self.ship_targets_rew = np.array([]).astype(int)

        self.yard_targets_pos = np.array([]).astype(int)
        self.yard_targets_rew = np.array([]).astype(int)

        # stats - remove
        self.total_bounties = 0
        self.conversions = 0
        self.total_loot = 0

        return

    def update(self, state):
        self.set_ship_targets(state)
        self.set_yard_targets(state)
        return

    def set_ship_targets(self, state):
        # we choose new targets from a pool of opponent ships. to select the
        # new targets we consider a score composed of "vulnerability" and
        # cargo. to measure vulnerability, we construct a graph where the
        # positions of our hunters have higher weights.
        weights = np.ones_like(state.sites)

        num_ships = state.my_ship_pos.size - fifos.fifo_pos.size
        num_targets = num_ships // SHIPS_PER_BOUNTY
        num_hunters = 5 * num_targets

        # find out which ships are likely to hunt others
        likely_hunters = np.argpartition(state.my_ship_hal, num_hunters - 1)
        likely_hunters = likely_hunters[0:num_hunters]
        hunters_pos = state.my_ship_pos[likely_hunters]

        if hunters_pos.size != 0:
            hood = state.dist[hunters_pos, :] <= 2
            weights += GRAPH_OPP_WEIGHT * np.sum(hood, axis=0)

        mean_weight = np.mean(weights)

        graph = targets.make_graph_csr(state, weights)

        # calculate position, halite, and vulnerability for all opponent ships
        # vulnerability is the ratio of distance to the nearest friendly yard
        # on the weighted graph over distance to the nearest friendly yard on
        # a graph with constant weights equal to mean_weight. a vulnerability
        # greater than one means we have hunters obstructing the path to the
        # nearest yard...
        opp_ship_pos = np.array([]).astype(int)
        opp_ship_hal = np.array([]).astype(int)
        opp_ship_vul = np.array([]).astype(int)
        opp_ship_dis = np.array([]).astype(int)

        for opp in state.opp_data.values():
            yards, ship_pos, ship_hal = opp[1:4]

            if yards.size == 0:
                ship_vul = 10 * np.ones_like(ship_pos)
                ship_dis = 10 * np.ones_like(ship_pos)
            else:
                graph_dist = dijkstra(graph, indices=yards, min_only=True)
                graph_dist = graph_dist[ship_pos]
                ship_dis = np.amin(state.dist[np.ix_(yards, ship_pos)], axis=0)
                ship_vul = (1 + graph_dist) / (1 + mean_weight * ship_dis)

            opp_ship_pos = np.append(opp_ship_pos, ship_pos)
            opp_ship_hal = np.append(opp_ship_hal, ship_hal)
            opp_ship_vul = np.append(opp_ship_vul, ship_vul)
            opp_ship_dis = np.append(opp_ship_dis, ship_dis)

        # nearby contains the number of hunters within distance 3
        # that also have strictly less cargo than the ship
        nearby = state.dist[np.ix_(state.my_ship_pos, opp_ship_pos)] <= 3
        less_hal = state.my_ship_hal[:, np.newaxis] < opp_ship_hal
        nearby = np.sum(nearby & less_hal, axis=0)

        # store current positions of previous targets that are still alive
        prev = np.array([val[0] for key, val in state.opp_ships.items()
                         if key in self.ship_targets]).astype(int)

        # get the indices of the ships that are already targeted
        # if a ship is too close to a friendly yard, it will probably escape
        # so we remove such ships from the targets
        # (note: & / | / ~ = and / or / not in numpy compatible way)
        target_bool = np.in1d(opp_ship_pos, prev)
        target_bool = target_bool & (opp_ship_dis >= 3)
        target_inds = np.flatnonzero(target_bool)

        # the pool of possible new targets consists of non-targeted ships
        # that are trapped (vulnerability > 1), have at least one hunter
        # nearby, and aren't too close to a friendly yard
        candidates = ~target_bool & (opp_ship_vul > 1)
        candidates = candidates & (opp_ship_dis >= 3) & (nearby >= 1)

        # we compute scores for each of the candidate ships indicating
        # the risk/reward of attacking them
        # make the scores of ships that are not candidates negative
        opp_ship_score = opp_ship_hal * opp_ship_vul
        opp_ship_score[~candidates] = -1

        # determine how many targets we would like to have and how many
        # new targets we should/can build. we only set new targets if
        # there is not a lot of halite left that we can mine
        ratio = np.sum(state.halite_map) / state.config.startingHalite

        if ratio > HUNTING_MAX_RATIO:
            num_new_targets = 0
        else:
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
        # we target the opponent whose score is closest to ours
        my_score = state.my_halite + np.sum(state.my_ship_hal)
        alive = lambda x: 0 < (state.opp_data[x][1].size
                             + state.opp_data[x][2].size)
        score = lambda x: alive(x) * (state.opp_data[x][0]
                                    + np.sum(state.opp_data[x][3]))
        score_diff = lambda x: np.abs(score(x) - my_score)
        closest = min(state.opp_data, key=score_diff)

        # depending on how many ships we have compared to others
        my_ships = np.setdiff1d(state.my_ship_pos, fifos.fifo_pos).size
        num_ships = lambda x: x[2].size
        max_opp_ships = max([num_ships(x) for x in state.opp_data.values()])

        # for most of the game, we only target yards if we have the most ships
        should_attack = my_ships > max_opp_ships

        # at the end of the game we target yards no matter what
        should_attack |= (state.total_steps - state.step) < YARD_HUNTING_FINAL

        # but stop attacking if we don't have a lot of ships anymore
        should_attack &= my_ships >= YARD_HUNTING_MIN_SHIPS

        # at the beginning of the game we never target yards
        should_attack &= state.step > YARD_HUNTING_START

        if should_attack:
            opp_yards, opp_ships = state.opp_data[closest][1:3]

            # in the final stage of the game we target all yards
            if state.total_steps - state.step < YARD_HUNTING_FINAL:
                self.yard_targets_pos = opp_yards
            # before that we only target unprotected yards
            else:
                self.yard_targets_pos = np.setdiff1d(opp_yards, opp_ships)

            # take 100 instead of 1000 here, so we prefer to target ships
            # and big halite cells
            self.yard_targets_rew = 100 * np.ones_like(self.yard_targets_pos)
        else:
            self.yard_targets_pos = np.array([]).astype(int)
            self.yard_targets_rew = np.array([]).astype(int)

        return

    def get_ship_targets(self, actor, state):
        pos, hal = state.my_ships[actor]

        # find targets that we can attack
        inds = np.flatnonzero(self.ship_targets_hal > hal)

        positions = np.array([]).astype(int)
        rewards = np.array([]).astype(int)

        # we put slightly lower bounties on the 4 sites adjacent to
        # the ship as well so that ships collapse on the target
        for pos, rew in zip(self.ship_targets_pos[inds],
                            self.ship_targets_rew[inds]):
            adj = np.flatnonzero(state.dist[pos, :] == 1)
            adj_rewards = (rew / 2) * np.ones_like(adj)
            positions = np.append(positions, adj)
            positions = np.append(positions, pos)
            rewards = np.append(rewards, adj_rewards)
            rewards = np.append(rewards, rew)

        return positions, rewards

    def get_yard_targets(self, actor, state):
        pos, hal = state.my_ships[actor]
        endgame = state.total_steps - state.step < YARD_HUNTING_FINAL

        # only hunt yards if we don't have any cargo or if its the
        # final phase of the game
        if hal > 0 and not endgame:
            return np.array([]).astype(int), np.array([]).astype(int)

        # at the end of the game we want most ships to go after yards
        # but during the bulk of the game we don't want to lose too
        # many ships due to shipyard hunting
        radius = YARD_HUNTING_RADIUS + endgame * 6
        inds = state.dist[self.yard_targets_pos, pos] <= radius

        return self.yard_targets_pos[inds], self.yard_targets_rew[inds]
