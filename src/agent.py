def agent(obs, config):
    # keep track of time spent during step
    tick = time()

    # parse (obs, config) into internal game state
    state = State(obs, config)

    # update the statistics we track across all turns
    memory.statistics(state)

    # initialize actions object
    actions = Actions(state)

    # convert appropriate ships into yards
    convert(state, actions)

    # determine which yards should spawn
    spawns = Spawns(state, actions)

    # place bounties on selected opponent ships/yards
    bounties = Bounties(state)

    # set preferences for where each ship would like to go
    targets = Targets(state, actions, bounties)

    # decide on moves for all ships/yards
    decide(state, actions, targets, spawns)

    # record time spent during this step
    tock = time()
    memory.total_time += tock - tick

    # print some statistics about the game before the last step
    if 2 + state.step == state.total_steps:
        memory.summary()

    # print how long the step took
    elif PRINT_STEP_TIMES:
        print(f"Step {1 + state.step} took {round(tock - tick, 2)} seconds")

    return actions.decided
