def decide(state, actions, targets, spawns):
    if len(actions.ships) != 0:
        # calculate the value per step of going to a specific site
        cost_matrix = np.vstack([move_row(ship, state, targets)
                                 for ship in actions.ships])

        # regularize infinities - replace them by very negative finite values
        # that can never be compensated by a good matching. so any matching
        # with an effective infinity is worse than any matching without one
        finite = np.isfinite(cost_matrix)
        eff_inf = np.sum(np.abs(cost_matrix[finite]))
        cost_matrix[~finite] = -eff_inf

        # find the optimal assignment of ships to destinations
        # the optimal assignment assignment assigns ship_inds[i] to
        # site_inds[i]
        ship_inds, sites = assignment(cost_matrix, maximize=True)

        for ship_ind, site in zip(ship_inds, sites):
            ship = actions.ships[ship_ind]

            pos, hal = state.my_ships[ship]

            move = state.pos_to_move(pos, site)

            # if doomed, move freely?

            if move is not None:
                actions.decided[ship] = move

            state.update(ship, move)


    while spawns.num_spawns > 0 and len(spawns.yards) > 0:
        yard = max(spawns.yards, key=spawns.yards.get)
        spawns.yards.pop(yard)
        actions.yards.remove(yard)
        if not state.moved_this_turn[state.my_yards[yard]]:
            actions.decided[yard] = "SPAWN"
            spawns.num_spawns -= 1

    return


def move_row(ship, state, targets):
    pos, hal = state.my_ships[ship]
    strict = True

    row = np.where(state.dist[pos, :] > 1, -np.inf, 0)

    row[targets.moves[ship]] = 10 * (5 - np.arange(5))

    none_flag = (targets.destinations[ship] not in state.my_yard_pos)
    none_flag = none_flag and (state.halite_map[pos] > 0)
    none_flag = none_flag and (pos != targets.destinations[ship])

    row[pos] -= 100 * none_flag

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
