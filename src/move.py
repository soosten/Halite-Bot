import numpy as np
from scipy.optimize import linear_sum_assignment as assignment

from settings import STEPS_SPIKE


def move(state, actions, targets):
    # if there are no ships pending, there is nothing to do
    if len(actions.ships) == 0:
        return

    # calculate the value per step of going to a specific site
    cost_matrix, threats, weak_threats = matrices(state, actions, targets)

    # regularize infinities - replace them by very negative finite values
    # that can never be compensated by a good matching. so any matching
    # with an effective infinity is worse than any matching without one
    finite = np.isfinite(cost_matrix)
    infinite = ~finite
    eff_inf = 1 + 2 * len(actions.ships) * np.max(np.abs(cost_matrix[finite]))
    cost_matrix[infinite] = -eff_inf

    # find the optimal assignment of ships to sites
    ship_inds, sites = assignment(cost_matrix, maximize=True)

    # go through the solution - if the assigned site is legal and safe
    # we move onto it, otherwise we add the ship to list of ships
    # for which decisions are made independently
    threatened = []
    depositing = []

    for ship_ind, site in zip(ship_inds, sites):
        ship = actions.ships[ship_ind]
        pos, hal = state.my_ships[ship]

        # if the ship was assigned to an unsafe site, decide on a move later
        if infinite[ship_ind, site] or threats[ship_ind, site]:
            threatened.append(ship_ind)
            continue

        # if the ship is depositing after the interest spike and
        # there is traffic at the yard, move freely
        spike = (state.total_steps - state.step) < STEPS_SPIKE
        spike = spike and (state.my_yard_pos.size > 0)
        if spike:
            yard_ind = np.argmin(state.dist[state.my_yard_pos, pos], axis=0)
            yard = state.my_yard_pos[yard_ind]
            close = state.dist[yard, pos] <= 3
            traffic = np.sum(state.dist[state.my_ship_pos, yard] <= 4) >= 10
            if traffic and close:
                depositing.append(ship_ind)
                continue

        decision = state.pos_to_move(pos, site)
        actions.decided[ship] = decision
        state.update(ship, decision)

    # decide on actions for ships that were assigned to unsafe sites
    for ship_ind in threatened:
        ship = actions.ships[ship_ind]
        pos, hal = state.my_ships[ship]

        illegal = (state.dist[pos, :] > 1)
        self_col = state.moved_this_turn
        weak_opp_col = weak_threats[ship_ind, :]

        # ideally, don't collide with opponents or our own ships
        exclude = illegal | self_col | weak_opp_col

        # if this is impossible, don't collide with opponents
        if np.sum(~exclude) == 0:
            exclude = illegal | weak_opp_col

        # if this is impossible, don't collide with our own ships
        if np.sum(~exclude) == 0:
            exclude = illegal | self_col

        # otherwise just make a move
        if np.sum(~exclude) == 0:
            exclude = illegal

        # in most of these situations, staying put is a exclude move
        # we could get lucky escaping other ships
        if np.sum(~exclude) >= 2:
            exclude[pos] = True

        # get the highest-ranked moved that is not excluded
        candidates = state.sites[~exclude]
        ranking = targets.moves[ship]
        ind = np.in1d(ranking, candidates).argmax()
        site = ranking[ind]

        decision = state.pos_to_move(pos, site)
        actions.decided[ship] = decision
        state.update(ship, decision)

    for ship_ind in depositing:
        ship = actions.ships[ship_ind]
        pos, hal = state.my_ships[ship]

        illegal = (state.dist[pos, :] > 1)
        weak_opp_col = weak_threats[ship_ind, :]

        # ideally don't collide with opponent ships
        exclude = illegal | weak_opp_col

        # otherwise just make a move
        if np.sum(~exclude) == 0:
            exclude = illegal

        # get the highest-ranked moved that is not excluded
        candidates = state.sites[~exclude]
        ranking = targets.moves[ship]
        ind = np.in1d(ranking, candidates).argmax()
        site = ranking[ind]

        decision = state.pos_to_move(pos, site)
        actions.decided[ship] = decision
        state.update(ship, decision)

    actions.ships.clear()
    return


def matrices(state, actions, targets):
    dims = (len(actions.ships), state.map_size ** 2)
    threat_matrix = np.zeros(dims, dtype=bool)
    weak_threat_matrix = np.zeros(dims, dtype=bool)
    cost_matrix = np.zeros(dims, dtype=float)

    # construct cost_matrix and threat_matrix
    for index in range(len(actions.ships)):
        ship = actions.ships[index]
        pos, hal = state.my_ships[ship]
        dest = targets.destinations[ship]

        yard_dist = np.amin(state.dist[state.my_yard_pos, pos], axis=0,
                            initial=state.map_size)
        strict = (yard_dist > 2) or (hal > 0)

        # find those ships that have less halite than we do
        # add 1 to hal if we want to have a strict halite comparison
        # since x < hal becomes x <= hal for integer values...
        ships = state.opp_ship_pos[state.opp_ship_hal < (hal + strict)]
        ship_dist = np.amin(state.dist[ships, :], axis=0,
                            initial=state.map_size)

        weak_ships = state.opp_ship_pos[state.opp_ship_hal < hal]
        weak_ship_dist = np.amin(state.dist[weak_ships, :], axis=0,
                                 initial=state.map_size)

        # threatened sites are opponent shipyard and sites where ships with
        # less cargo can be in one step
        threat_matrix[index, :] = np.in1d(state.sites, state.opp_yard_pos)
        threat_matrix[index, :] |= (ship_dist <= 1)

        weak_threat_matrix[index, :] = np.in1d(state.sites, state.opp_yard_pos)
        weak_threat_matrix[index, :] |= (weak_ship_dist <= 1)

        # turn off opponent collision checking if we are hunting a yard
        # and are close enough to strike
        if (dest in state.opp_yard_pos) and (state.dist[pos, dest] <= 2):
            threat_matrix[index, :] = False
            weak_threat_matrix[index, :] = False

        # penalize legal moves by ranking from targets
        cost_matrix[index, targets.moves[ship]] = -10 * np.arange(5)

        # try not to pick up unnecessary cargo that makes ships vulnerable
        # so add an additional penalty to the current position
        no_cargo = (dest not in state.my_yard_pos)
        no_cargo = no_cargo and (state.halite_map[pos] > 0)
        no_cargo = no_cargo and (pos != dest)
        cost_matrix[index, pos] -= (100 if no_cargo else 0)

        # penalize going to unsafe squares
        cost_matrix[index, threat_matrix[index, :]] -= 1000

        # give higher priority to ships with higher cargo, but highest
        # priority to ships with no cargo at all
        if hal == 0:
            multiplier = 3
        else:
            rank = np.sum(hal > state.my_ship_hal)
            multiplier = 1 + rank / state.my_ship_hal.size

        # multiplier = 1 + np.sum(value > ship_values) / ship_values.size
        cost_matrix[index, :] = multiplier * cost_matrix[index, :]

        # penalize illegal moves with infinity
        cost_matrix[index, state.dist[pos, :] > 1] = -np.inf

    return cost_matrix, threat_matrix, weak_threat_matrix
