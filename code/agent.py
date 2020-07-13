def agent(obs, config):
    # internal game state, to be updated as we decide on actions
    state = State(obs, config)

    # list of ships/yards for which we need to decide on an action
    queue = Queue(state)

    # as we decide on actions for our ships/yards we write them into the
    # actions dictionary, which is what is returned to the environment
    # we initialize actions with the minimal actions needed to ensure
    # survival (usual just the empty dictionary {})
    actions = survive(state, queue)

    # update which yards should operate under fifo system
    # and strip any ships on fifo yards from the queue
    global fifos
    fifos.update(state)
    fifos.strip(state, queue)

    # update any special targets for our ships such as opponent
    # ships/yards that should be targeted by our hunters
    global targets
    targets.update(state)

    # now decide on "normal" actions for the remaining actors
    while queue.pending():
        # schedule the next ship/yard
        actor = queue.schedule(state)

        # decide on an action for it
        action = decide(state, actor)

        # update game state with consequence of action
        state.update(actor, action)

        # put any ships on fifo yards back in the queue if
        # the action resulted in a new ship on a fifo yard
        fifos.resolve(state, queue, actor, action)

        # write action into dictionary of actions to return
        if action is not None:
            actions[actor] = action

    # update the global statistics we track across all turns
    global stats
    stats.update(state)

    print(1 + state.step)
    if state.step == 398:
        print(f"converted {targets.conversions} / {targets.total_bounties}")
        print(f"total: {targets.total_loot}")

    return actions
