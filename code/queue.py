class Queue:
    def __init__(self, state):
        self.ships = list(state.my_ships.keys())
        self.yards = list(state.my_yards.keys())
        return

    def pending(self):
        return len(self.ships) + len(self.yards) > 0

    def remove(self, actor):
        if actor in self.ships:
            self.ships.remove(actor)
        if actor in self.yards:
            self.yards.remove(actor)
        return

    def schedule(self, state):
        # if there are ships, schedule the ship with the highest priority
        if len(self.ships) > 0:
            nextup = max(self.ships, key=lambda s: self.priority(state, s))

        # else we have only yards left. take yards with small clusters first
        # so ships will spawn where there is the most space
        else:
            clust_size = lambda y: state.get_cluster(state.my_yards[y]).size
            nextup = min(self.yards, key=clust_size)

        # pop the scheduled actor from the pending list
        self.remove(nextup)
        return nextup

    def priority(self, state, ship):
        # ships have highest priority if they are likely to convert
        if should_convert(ship, state):
            return 2000

        # next-highest priority if they have one or less non-colliding moves
        moves = [None, "NORTH", "SOUTH", "EAST", "WEST"]
        collision = lambda m: state.opp_collision(ship, m) \
            or state.self_collision(ship, m)
        if len([1 for move in moves if not collision(move)]) <= 1:
            return 1000

        # otherwise they are ranked by cargo
        return state.my_ships[ship][1]
