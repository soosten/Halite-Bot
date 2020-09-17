import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from settings import (SHIPS_PER_BOUNTY, HUNTING_MAX_RATIO, HUNTING_STEP,
                      YARD_HUNTING_FINAL, YARD_HUNTING_MIN_SHIPS, HUNT_WEIGHT,
                      YARD_HUNTING_START, YARD_HUNTING_RADIUS, HUNT_RADIUS,
                      HUNT_NEARBY, STEPS_SPIKE)


class Bounties:
    def __init__(self, state, ship_target_memory):
        self.set_ship_targets(state, ship_target_memory)
        self.set_yard_targets(state)
        return

    def set_ship_targets(self, state, ship_target_memory):
        # set number of new targets depending on how many ships we have
        self.num_targets = state.my_ship_pos.size // SHIPS_PER_BOUNTY

        # we choose new targets from a pool of opponent ships. to select the
        # new targets we consider a score composed of "vulnerability" and
        # cargo. to measure vulnerability, we construct a graph where the
        # positions of our hunters have higher weights.
        graph = self.make_graph(state)

        # calculate position, halite, and vulnerability for all opponent ships
        # vulnerability is the ratio of distance to the nearest friendly yard
        # on the weighted graph over distance to the nearest friendly yard on
        # a graph with constant weights equal to mean_weight. a vulnerability
        # greater than one means we have hunters obstructing the path to the
        # nearest yard...
        opp_ship_pos = np.array([], dtype=int)
        opp_ship_hal = np.array([], dtype=int)
        opp_ship_vul = np.array([], dtype=float)
        opp_ship_dis = np.array([], dtype=int)

        for opp in state.opp_data.values():
            yards, ship_pos, ship_hal = opp[1:4]

            if yards.size == 0:
                ship_vul = np.full_like(ship_pos, 10)
                ship_dis = np.full_like(ship_pos, 10)
            else:
                graph_dist = dijkstra(graph, indices=yards, min_only=True)
                graph_dist = graph_dist[ship_pos]
                ship_dis = np.amin(state.dist[np.ix_(yards, ship_pos)], axis=0)
                ship_vul = (1 + graph_dist) / (1 + ship_dis)

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
                         if key in ship_target_memory], dtype=int)

        # get the indices of the ships that are already targeted
        # if a ship is too close to a friendly yard, it will probably escape
        # so we remove such ships from the targets
        target_bool = np.in1d(opp_ship_pos, prev) & (opp_ship_dis >= 3)
        target_inds = np.flatnonzero(target_bool)

        # the pool of possible new targets consists of non-targeted ships
        # that are trapped (vulnerability > 1), have at least one hunter
        # nearby, and aren't too close to a friendly yard
        candidates = ~target_bool & (opp_ship_vul > 1)
        candidates &= (opp_ship_dis >= 3) & (nearby >= HUNT_NEARBY)

        # we compute scores for each of the candidate ships indicating
        # the risk/reward of attacking them
        # make the scores of ships that are not candidates negative
        opp_ship_score = opp_ship_hal * opp_ship_vul
        opp_ship_score[~candidates] = -1

        # determine how many targets we would like to have and how many
        # new targets we should/can build. we only set new targets if
        # there is not a lot of halite left that we can mine - however
        # we always hunt after the first part of the game
        ratio = np.sum(state.halite_map) / state.starting_halite
        if (ratio > HUNTING_MAX_RATIO) and (state.step < HUNTING_STEP):
            num_new_targets = 0
        else:
            num_new_targets = max(self.num_targets - target_inds.size, 0)
            num_new_targets = min(num_new_targets, np.sum(candidates))

        # we can take those num_new_targets ships with maximum score
        # since scores are >= 0  and we forced the scores of non-candidate
        # ships to equal -1. see comment before argpartition in set_hunters
        new_inds = np.argpartition(-opp_ship_score, num_new_targets - 1)
        target_inds = np.append(target_inds, new_inds[0:num_new_targets])

        # set position/halite/rewards for the targets
        self.ship_targets_pos = opp_ship_pos[target_inds]
        self.ship_targets_hal = opp_ship_hal[target_inds]
        rew = state.max_halite
        self.ship_targets_rew = np.full_like(self.ship_targets_pos, rew)

        # write the new targets in the ship_targets list
        self.target_list = [key for key, val in state.opp_ships.items()
                            if val[0] in self.ship_targets_pos]

        return

    def set_yard_targets(self, state):
        # always attack any yards that are too close to our own
        # inds = np.ix_(state.my_yard_pos, state.opp_yard_pos)
        # dists = np.min(state.dist[inds], axis=0, initial=state.map_size)
        # self.yard_targets_pos = state.opp_yard_pos[dists <= 3]

        # we target the opponent whose score is closest to ours
        my_score = state.my_halite + np.sum(state.my_ship_hal)
        score_diff = lambda opp: abs(state.opp_scores[opp] - my_score)
        closest = min(state.opp_scores, key=score_diff, default=None)

        # depending on how many ships we have compared to others
        my_ships = state.my_ship_pos.size
        max_opp_ships = max(state.opp_num_ships.values(), default=0)

        # attack yards at the end of the game
        should_attack = (state.total_steps - state.step) < YARD_HUNTING_FINAL

        # or if we have a lot of ships
        should_attack = should_attack or (my_ships > max_opp_ships)

        # but stop attacking if we don't have a lot of ships anymore
        should_attack = should_attack and (my_ships >= YARD_HUNTING_MIN_SHIPS)

        # and don't attack if its too early in the game
        should_attack = should_attack and (state.step > YARD_HUNTING_START)

        if should_attack and (closest is not None):
            opp_yards, opp_ships = state.opp_data[closest][1:3]

            # in the final stage of the game we target all yards
            # before that we only target unprotected yards
            if state.total_steps - state.step > YARD_HUNTING_FINAL:
                opp_yards = np.setdiff1d(opp_yards, opp_ships)

            self.yard_targets_pos = opp_yards
        else:
            self.yard_targets_pos = np.array([], dtype=int)

        # take 100 instead of 500 here, so we prefer to target ships
        # and big halite cells
        rew = state.max_halite / 5
        self.yard_targets_rew = np.full_like(self.yard_targets_pos, rew)
        return

    def get_ship_targets(self, actor, state):
        # stop hunting ships after the interest rate spike
        if state.total_steps - state.step < STEPS_SPIKE:
            return np.array([], dtype=int), np.array([], dtype=int)

        pos, hal = state.my_ships[actor]

        # find targets that we can attack
        attackable = self.ship_targets_hal > hal
        targets = self.ship_targets_pos[attackable]
        rewards = self.ship_targets_rew[attackable]

        full_pos = np.array([], dtype=int)
        full_rew = np.array([], dtype=int)

        # we put slightly lower bounties on the 4 sites adjacent to
        # the ship as well so that ships collapse on the target
        for pos, rew in zip(targets, rewards):
            adj = np.flatnonzero(state.dist[pos, :] == 1)
            adj_rewards = np.full_like(adj, rew)
            full_pos = np.append(full_pos, adj)
            full_pos = np.append(full_pos, pos)
            full_rew = np.append(full_rew, adj_rewards)
            full_rew = np.append(full_rew, 2 * rew)

        # remove any duplicate indices and rewards
        full_pos, inds = np.unique(full_pos, return_index=True)
        full_rew = full_rew[inds]

        return full_pos, full_rew

    def get_yard_targets(self, actor, state):
        pos, hal = state.my_ships[actor]
        endgame = state.total_steps - state.step < YARD_HUNTING_FINAL

        # only hunt yards if we don't have any cargo or if its the
        # final phase of the game
        if hal > 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # at the end of the game we want most ships to go after yards
        # but during the bulk of the game we don't want to lose too
        # many ships due to shipyard hunting
        radius = (YARD_HUNTING_RADIUS + 6) if endgame else YARD_HUNTING_RADIUS
        inds = state.dist[self.yard_targets_pos, pos] <= radius

        return self.yard_targets_pos[inds], self.yard_targets_rew[inds]

    def make_graph(self, state):
        nsites = state.map_size ** 2
        weights = np.ones(nsites)

        # find out which ships are likely to hunt others and weight
        # the graph in a neighborhood of their positions
        num_hunters = 5 * self.num_targets
        likely_hunters = np.argpartition(state.my_ship_hal, num_hunters - 1)
        likely_hunters = likely_hunters[0:num_hunters]
        hunters_pos = state.my_ship_pos[likely_hunters]
        hood = state.dist[hunters_pos, :] <= HUNT_RADIUS
        weights += HUNT_WEIGHT * np.sum(hood, axis=0, initial=0)

        # weight at any edge (x,y) is (w[x] + w[y])/2
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
