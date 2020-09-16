import numpy as np

from settings import (YARD_SCHEDULE, YARD_MAX_STEP, MIN_OPP_YARD_DIST,
                      MIN_YARD_DIST, SUPPORT_DIST, STEPS_INITIAL)


def convert(state, actions):
    # if we don't have any yards, we try to convert the ship
    # with the most cargo immediately
    if len(state.my_yards) == 0:
        cargo = lambda ship: state.my_ships[ship][1]
        ship = max(actions.ships, key=cargo)

        if legal(ship, state):
            actions.decided[ship] = "CONVERT"
            state.update(ship, "CONVERT")
            actions.ships.remove(ship)

        return

    # otherwise, we convert a ship if we have too few yards for our ships
    # and it is not too late in the game, provided we have ships
    yards = working_yards(state)
    yards_wanted = sum([x <= state.my_ship_pos.size for x in YARD_SCHEDULE])
    should_convert = (yards.size < yards_wanted)
    should_convert = should_convert and (state.step < YARD_MAX_STEP)
    should_convert = should_convert and (len(actions.ships) > 0)

    if not should_convert:
        return

    supports = np.sum(state.dist[yards, :] <= SUPPORT_DIST, axis=0, initial=0)
    triangles = (supports >= min(yards.size, 2))

    closest = np.amin(state.dist[yards, :], axis=0, initial=state.map_size)
    triangles &= (closest >= MIN_YARD_DIST)

    opp_yard_dist = np.amin(state.dist[state.opp_yard_pos, :], axis=0,
                           initial=state.map_size)
    triangles &= (opp_yard_dist >= MIN_OPP_YARD_DIST)

    opp_ship_dist = np.amin(state.dist[state.opp_ship_pos, :], axis=0,
                           initial=state.map_size)
    triangles &= (opp_ship_dist >= 2)

    convertable = state.my_ship_pos[triangles[state.my_ship_pos]]

    if convertable.size == 0:
        return

    halite = np.flatnonzero(state.halite_map > 0)
    hood = state.dist[np.ix_(halite, convertable)] <= 6
    cells = np.sum(hood, axis=0, initial=0)

    pos = lambda ship: state.my_ships[ship][0]
    pos_to_ship = {pos(ship): ship for ship in actions.ships}
    ship_pos = convertable[cells.argmax()]
    ship = pos_to_ship.get(ship_pos, None)

    if (ship is not None) and legal(ship, state):
        actions.decided[ship] = "CONVERT"
        state.update(ship, "CONVERT")
        actions.ships.remove(ship)

    return


# returns True if CONVERT is a legal action for ship - need to have enough
# halite and not be on another yard. if you only have one ship, you can
# only convert if you still have enough halite to spawn a ship afterwards
def legal(ship, state):
    pos, hal = state.my_ships[ship]
    minhal = state.convert_cost - hal
    if len(state.my_ships) == 1:
        minhal += state.spawn_cost
    return (state.my_halite >= minhal) and (pos not in state.my_yard_pos)


def working_yards(state):
    if state.step > STEPS_INITIAL:
        inds = np.ix_(state.opp_yard_pos, state.my_yard_pos)
        dists = np.amin(state.dist[inds], axis=0, initial=state.map_size)
        return state.my_yard_pos[dists > 2]
    else:
        return state.my_yard_pos
