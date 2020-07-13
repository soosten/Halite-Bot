class Stats:
    def __init__(self):
        self.last_state = None
        self.yard_attacks = False
        return

    def update(self, state):
        if self.last_state is None:
            self.last_state = state
            return

        # check if any of the players lost a yard in the last turn
        # this works because if a yard gets destroyed, it takes at least
        # one turn to move a new ship there before it can convert
        opp_diff = np.setdiff1d(self.last_state.opp_yard_pos,
                                state.opp_yard_pos)
        my_diff = np.setdiff1d(self.last_state.my_yard_pos, state.my_yard_pos)

        if opp_diff.size + my_diff.size > 0:
            self.yard_attacks = True

        self.last_state = state
        return
