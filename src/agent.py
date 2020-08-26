def agent(obs, config):
    # read (obs, config) into internal game state object
    state = State(obs, config)

    # update statistics we track across all turns and store total cargo at
    # the beginning of the turn before any calls to state.update()
    memory.statistics(state)
    memory.cargo = np.sum(state.my_ship_hal)

    # actions object stores a list of pending ships/yards. as we decide on
    # actions, we remove the ships/yards from the pending lists and store
    # them in a dictionary together with their actions
    actions = Actions(state)

    # convert appropriate ships into yards
    convert(state, actions)

    # place bounties on selected opponent ships/yards
    bounties = Bounties(state)

    # set destinations for ships and rank moves by how much closer
    # we get to the destinations
    targets = Targets(state, actions, bounties)

    # decide on moves for ships
    move(state, actions, targets)

    # decide which yards should spawn
    spawn(state, actions)

    # print some statistics about the game before the last step
    if 2 + state.step == state.total_steps:
        memory.summary()

    return actions.decided
