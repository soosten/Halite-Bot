def conversions(state, queue, actions):
    # cannot convert if there are no ships
    if len(queue.ships.keys()) == 0:
        return

    # don't build any yards in the final phase of the game
    # survive() already ensures we keep at least one
    if state.step > YARD_MAX_STEP:
        return False

    # otherwise keep a number of yards depending on how many ships we have
    num_ships = state.my_ship_pos.size - fifos.fifo_pos.size
    yards_wanted = np.sum(num_ships > YARD_SCHEDULE)
    if state.my_yard_pos.size >= yards_wanted:
        return

    # the score of each ship is the number of halite cells within YARD_RADIUS
    # the score defaults to 0 if there is another yarc within YARD_DIST or
    # if there are less than MIN_CELLS halite cells nearby
    def score(ship):
        # compute distance to nearest yard
        pos = state.my_ships[ship][0]
        yard_dist = np.amin(state.dist[state.my_yard_pos, pos])

        # count number of halite cells within YARD_RADIUS
        hood = (state.dist[pos, :] <= YARD_RADIUS) & (state.halite_map > 0)
        cells = np.count_nonzero(hood)

        # score is number of halite cells unless there are too few
        # or we are too close to a yard
        return cells * (yard_dist >= YARD_DIST and cells >= MIN_CELLS)

    # convert the ship with maximum score
    ship = max(queue.ships.keys(), key=score)

    # but only if conversion is legal, does not result in collision
    # and the ship has a non-zero score
    legal = "CONVERT" in state.legal_actions(ship)
    collision = state.self_collision(ship, None)
    collision = collision or state.opp_collision(ship, None)

    if score(ship) > 0 and legal and not collision:
        queue.remove(ship)
        state.update(ship, "CONVERT")
        actions[ship] = "CONVERT"

    return
