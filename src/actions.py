class Actions:
    def __init__(self, state):
        self.decided = {}
        self.ships = list(state.my_ships)
        self.yards = list(state.my_yards)
        return

    def asdict(self):
        return {k: v for k, v in self.decided.items() if v is not None}
