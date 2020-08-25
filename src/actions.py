class Actions:
    def __init__(self, state):
        self.decided = {}
        self.ships = list(state.my_ships)
        self.yards = list(state.my_yards)

        return
