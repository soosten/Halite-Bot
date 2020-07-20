class Stats:
    def __init__(self):
        self.last_state = None
        self.state = None
        self.yard_attackers = []
        return

    def update(self, argstate):
        # on the first turn, just copy the state into last_state and return
        if self.last_state is None:
            self.last_state = deepcopy(argstate)
            return

        # use deepcopy that we keep the state at the beginning of our turn
        # and don't update as we go through deciding actions for our actors
        self.state = deepcopy(argstate)

        # determine if anyone destroyed a shipyard last turn and
        # add the suspect to list of yard attackers
        self.find_yard_attacks()

        # save the current state as the previous state
        self.last_state = self.state
        return

    def find_yard_attacks(self):
        # store yard positions for all players for this and last state
        last_all_yard_pos = np.union1d(self.last_state.my_yard_pos,
                                       self.last_state.opp_yard_pos)

        all_yard_pos = np.union1d(self.state.my_yard_pos,
                                  self.state.opp_yard_pos)

        # for every opponent we check whether one of the yards of the
        # other three players has been destroyed during the last turn
        # if one of the opponent ships was next to the destroyed yard
        # on the previous turn, we add that opponent to the list of
        # yard attackers. the check isn't foolproof if there are two
        # opponents near a yard.
        for opp in self.last_state.opp_data:
            if opp in self.yard_attackers:
                continue

            last_yard_pos, last_ship_pos = self.last_state.opp_data[opp][1:3]

            yard_pos = self.state.opp_data[opp][1]

            last_other_yards = np.setdiff1d(last_all_yard_pos, last_yard_pos)
            other_yards = np.setdiff1d(all_yard_pos, yard_pos)

            # this works because if a yard gets destroyed, it takes at least
            # one turn to move a new ship there before it can convert
            destroyed = np.setdiff1d(last_other_yards, other_yards)

            suspects = self.state.dist[np.ix_(destroyed, last_ship_pos)] == 1
            if np.sum(suspects) > 0:
                self.yard_attackers.append(opp)

        return
