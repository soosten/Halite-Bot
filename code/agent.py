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

    # decide if any ships should convert
    conversions(state, queue, actions)

    # update which yards should operate under fifo system
    # and strip any ships on fifo yards from the queue
    fifos.update(state)
    fifos.strip(state, queue)

    # update any special targets for our ships such as opponent
    # ships/yards that should be targeted by our hunters
    bounties.update(state)

    # set preferences for where each ship would like to go
    targets.calculate(state, queue)

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
    stats.update(state)

    if state.step == 398:
        print(f"{bounties.conversions} / {bounties.total_bounties} = {round(bounties.conversions / (0.01 + bounties.total_bounties), 2)}")
        print(f"{bounties.total_loot}")

    return actions
