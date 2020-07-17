class Queue:
    def __init__(self, state):
        self.ships = list(state.my_ships.keys())
        self.yards = list(state.my_yards.keys())

        # scheduling flags - if conv_flag is raised, we no longer check for
        # ships that should convert at the beginning of schedule(), etc. this
        # way we don't run the expensive loops checking for this as often
        # these were causing the bot to time out in competitions on kaggle.com
        self.fifo_flag = False
        self.move_flag = False
        self.conv_flag = False
        return

    def pending(self):
        return (len(self.ships) + len(self.yards)) > 0

    def remove(self, actor):
        if actor in self.ships:
            self.ships.remove(actor)

        if actor in self.yards:
            self.yards.remove(actor)

        return

    def schedule(self, state):
        nextup = None

        if len(self.ships) > 0:
            # first schedule all ships that are likely to convert
            if not self.conv_flag:
                nextup = next((ship for ship in self.ships if
                               should_convert(ship, state)), None)

            # then schedule those with <= 1 possible moves
            # if this code exectutes, there were no ships that should
            # convert so we raise conv_flag to stop checking in the future
            if nextup is None and not self.move_flag:
                self.conv_flag = True
                nextup = next((ship for ship in self.ships if
                               state.num_moves(ship) <= 1), None)

            # then schedule remaining ships by cargo
            # if this code exectutes, there were no ships that had only one
            # move so we raise move_flag to stop checking in the future
            # this isn't foolproof but should not cause collision very often
            if nextup is None:
                self.move_flag = True
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
                clsize = lambda y: state.get_cluster(state.my_yards[y]).size
                nextup = min(self.yards, key=clsize)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup
