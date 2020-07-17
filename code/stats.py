class Stats:
    def __init__(self):
        self.last_state = None
        self.current_state = None

        self.yard_attackers = []
        self.yard_attacks = False

        self.idling_ship_pos = np.array([]).astype(int)
        self.snapshots = []
        return

    def update(self, state):
        if self.last_state is None:
            self.last_state = state
            return

        self.current_state = state

        # check if there are any idling ships
        self.set_idling_ships()

        self.find_yard_attacks()

        # check if any of the players lost a yard in the last turn
        # this works because if a yard gets destroyed, it takes at least
        # one turn to move a new ship there before it can convert
        opp_diff = np.setdiff1d(self.last_state.opp_yard_pos,
                                state.opp_yard_pos)
        my_diff = np.setdiff1d(self.last_state.my_yard_pos, state.my_yard_pos)

        if opp_diff.size + my_diff.size > 0:
            self.yard_attacks = True

        self.last_state = self.current_state
        return

    def set_idling_ships(self):
        return

    def find_yard_attacks(self):
        return
