def agent(obs, config):
    # read (obs, config) into internal game state object
    state = State(obs, config)

    # update statistics we track across all turns
    memory.statistics(state)

    # update which shipyard should be protected
    memory.protection(state)

    # actions object stores a list of pending ships/yards. as we decide on
    # actions, we remove the ships/yards from the pending lists and store
    # them in a dictionary together with their actions
    actions = Actions(state)

    # convert appropriate ships into yards
    convert(state, actions)

    # plan where we want to spawn new ships
    spawns = Spawns(state, actions)

    # place bounties on selected opponent ships/yards
    bounties = Bounties(state)

    # set destinations for ships and rank moves by how much closer
    # we get to the destinations
    targets = Targets(state, actions, bounties, spawns)

    # decide on moves for ships
    move(state, actions, targets)

    # spawn the new ships at unoccupied shipyards
    spawns.spawn(state, actions)

    # print some statistics about the game before the last step
    if 2 + state.step == state.total_steps:
        memory.summary()

    return actions.asdict()
