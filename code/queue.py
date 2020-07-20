class Queue:
    def __init__(self, state):
        # store ships with list of sites they can go to without colliding
        # store yards with size of the cluster they are part of
        self.ships = {ship: state.safe_sites(ship) for ship in state.my_ships}
        self.yards = {yard: state.get_cluster(yard_pos).size
                      for yard, yard_pos in state.my_yards.items()}

        # scheduling flags - if conv_flag is raised, we no longer check for
        # ships that should convert at the beginning of schedule(), etc. this
        # way we don't run the expensive loops checking for this as often
        # these were causing the bot to time out on kaggle
        self.fifo_flag = False
        self.conv_flag = False
        return

    def pending(self):
        return (len(self.ships) + len(self.yards)) > 0

    def remove(self, actor):
        self.ships.pop(actor, None)
        self.yards.pop(actor, None)
        return

    def schedule(self, state):
        # update the non-colliding mvoes for each ship with
        # the result of the last turn
        self.ships = {key: np.setdiff1d(val, state.moved_this_turn)
                      for key, val in self.ships.items()}

        nextup = None

        if len(self.ships) > 0:
            # first schedule all ships that are likely to convert
            if not self.conv_flag:
                nextup = next((ship for ship in self.ships if
                               should_convert(ship, state)), None)

            # then schedule those with <= 1 possible moves
            # if this code exectutes, there were no ships that should
            # convert so we raise conv_flag to stop checking in the future
            if nextup is None:
                self.conv_flag = True
                nextup = next((ship for ship, val in self.ships.items() if
                               val.size <= 1), None)

            # finally schedule remaining ships by cargo
            if nextup is None:
                cargo = lambda ship: state.my_ships[ship][1]
                nextup = max(self.ships, key=cargo)

        else:
            # first schedule any fifo yards so ships will spawn at
            # empty fifo yards first
            if not self.fifo_flag:
                nextup = next((yard for yard in self.yards if
                               state.my_yards[yard] in fifos.fifo_pos), None)

            # then schedule yards by the size of their cluster so ships
            # spawn where there is the most room. if this code executes
            # all the fifo yard have been scheduled so we raise fifo_flag
            # to stop checking in the future
            if nextup is None:
                self.fifo_flag = True
                nextup = min(self.yards, key=self.yards.get)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup
