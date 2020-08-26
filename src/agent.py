def agent(obs, config):
    tick = time()  # remove

    # parse (obs, config) into internal game state
    state = State(obs, config)

    # update the statistics we track across all turns
    memory.statistics(state)
    memory.cargo = np.sum(state.my_ship_hal)

    # initialize actions object
    actions = Actions(state)

    # convert appropriate ships into yards
    convert(state, actions)

    # place bounties on selected opponent ships/yards
    bounties = Bounties(state)

    # set preferences for where each ship would like to go
    targets = Targets(state, actions, bounties)

    # decide on moves for ships
    move(state, actions, targets)

    # decide which yards should spawn
    spawn(state, actions)

    # print some statistics about the game before the last step
    if 2 + state.step == state.total_steps:
        memory.summary()

    else:  # remove
        tock = time()
        print(f"Step {1 + state.step} took {round(tock - tick, 2)} seconds")

    return actions.decided
