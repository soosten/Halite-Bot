def move(state, actions, targets):
    # if there are no ships pending, there is nothing to do
    if len(actions.ships) == 0:
        return

    # calculate the value per step of going to a specific site
    cost_matrix, threat_matrix, infinite = matrices(state, actions, targets)

    # find the optimal assignment of ships to sites
    ship_inds, sites = assignment(cost_matrix, maximize=True)

    # go through the solution - if the assigned site is legal and safe
    # we move onto it, otherwise we add the ship to list of ships
    # for which decisions are made independently
    ind_to_ship = actions.ships.copy()

    for ship_ind, site in zip(ship_inds, sites):
        ship = ind_to_ship[ship_ind]

        # if doomed, move freely?
        if infinite[ship_ind, site] or threat_matrix[ship_ind, site]:
            pass

        pos, hal = state.my_ships[ship]
        decision = state.pos_to_move(pos, site)

        if decision is not None:
            actions.decided[ship] = decision

        actions.ships.remove(ship)
        state.update(ship, decision)

    return


def matrices(state, actions, targets):
    dims = (len(actions.ships), state.map_size ** 2)
    threat_matrix = np.empty(dims, dtype=np.bool_)
    cost_matrix = np.empty(dims, dtype=np.float_)

    # construct cost_matrix and threat_matrix
    for index in range(len(actions.ships)):
        ship = actions.ships[index]
        pos, hal = state.my_ships[ship]
        dest = targets.destinations[ship]
        value = targets.values[ship]

        yard_dist = np.amin(state.dist[state.my_yard_pos, pos], axis=0,
                            initial=state.map_size)
        strict = (yard_dist > 2) or (hal > 0)

        # find those ships that have less halite than we do
        # add 1 to hal if we want to have a strict halite comparison
        # since x < hal becomes x <= hal for integer values...
        ships = state.opp_ship_pos[state.opp_ship_hal < (hal + strict)]
        ship_dist = np.amin(state.dist[ships, :], axis=0, initial=state.map_size)

        # threatened sites are opponent shipyard and sites where ships with
        # less cargo can be in one step
        threat_matrix[index, :] = np.in1d(state.sites, state.opp_yard_pos)
        threat_matrix[index, :] |= (ship_dist <= 1)

        # penalize illegal moves with infinity
        cost_matrix[index, :] = np.where(state.dist[pos, :] > 1, -np.inf, 0)

        # penalize legal moves by ranking from targets
        cost_matrix[index, targets.moves[ship]] = -10 * np.arange(5)

        # try not to pick up unnecessary cargo that makes ships vulnerable
        # so add an additional penalty to the current position
        no_cargo = (dest not in state.my_yard_pos)
        no_cargo = no_cargo and (state.halite_map[pos] > 0)
        no_cargo = no_cargo and (pos != dest)
        cost_matrix[index, pos] -= 100 * no_cargo

        # turn off opponent collision checking if we are hunting a yard
        # and are close enough to strike
        if (dest not in state.opp_yard_pos) or (state.dist[pos, dest] > 2):
            cost_matrix[index, :] -= 1000 * threat_matrix[index, :]

        cost_matrix[index, :] = value * cost_matrix[index, :]

    # regularize infinities - replace them by very negative finite values
    # that can never be compensated by a good matching. so any matching
    # with an effective infinity is worse than any matching without one
    finite = np.isfinite(cost_matrix)
    infinite = ~finite
    eff_inf = 1 + len(actions.ships) * np.max(np.abs(cost_matrix[finite]))
    cost_matrix[infinite] = -eff_inf

    return cost_matrix, threat_matrix, infinite
