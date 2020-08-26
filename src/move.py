def move(state, actions, targets):
    if len(actions.ships) == 0:
        return

    # calculate the value per step of going to a specific site
    cost_matrix = np.vstack([move_row(ship, state, targets)
                             for ship in actions.ships])

    # regularize infinities - replace them by very negative finite values
    # that can never be compensated by a good matching. so any matching
    # with an effective infinity is worse than any matching without one
    finite = np.isfinite(cost_matrix)
    eff_inf = 1 + len(actions.ships) * np.max(np.abs(cost_matrix[finite]))
    cost_matrix[~finite] = -eff_inf

    # find the optimal assignment of ships to destinations
    # the optimal assignment assignment assigns ship_inds[i] to
    # site_inds[i]
    ship_inds, sites = assignment(cost_matrix, maximize=True)

    for ship_ind, site in zip(ship_inds, sites):
        ship = actions.ships[ship_ind]

        pos, hal = state.my_ships[ship]

        decision = state.pos_to_move(pos, site)

        # if doomed, move freely?

        if decision is not None:
            actions.decided[ship] = decision

        state.update(ship, decision)

    return


def move_row(ship, state, targets):
    pos, hal = state.my_ships[ship]
    dest = targets.destinations[ship]

    # penalize illegal moves with infinity
    row = np.where(state.dist[pos, :] > 1, -np.inf, 0)

    # penalize legal moves by ranking from targets
    row[targets.moves[ship]] = -10 * np.arange(5)

    # try not to pick up unnecessary cargo that makes ships vulnerable
    # so add an additional penalty to the current position
    no_cargo = (dest not in state.my_yard_pos)
    no_cargo = no_cargo and (state.halite_map[pos] > 0)
    no_cargo = no_cargo and (pos != dest)
    row[pos] -= 100 * no_cargo


    # turn off opponent collision checking if we are hunting a yard
    # and are close enough to strike
    if (dest not in state.opp_yard_pos) or (state.dist[pos, dest] > 2):
        # no undesired opponent collisions - check strictly if the ship has
        # cargo or is far from the yard. hopefully, this will make ships less
        # locked down if some attackers are waiting in front of our yard
        strict = np.amin(state.dist[state.my_yard_pos, pos], axis=0) > 2
        strict = strict or (hal > 0)

        # find those ships that have less halite than we do
        # add 1 to hal if we want to have a strict halite comparison
        # since x < hal becomes x <= hal for integer values...
        threats = state.opp_ship_pos[state.opp_ship_hal < (hal + strict)]

        # set of sites where ships with less cargo can be in one step
        threat_dist = np.amin(state.dist[threats, :], axis=0,
                              initial=state.map_size)
        unsafe = (threat_dist <= 1)

        # opponent yards
        unsafe |= np.in1d(state.sites, state.opp_yard_pos)

        row -= 1000 * unsafe

    return row
