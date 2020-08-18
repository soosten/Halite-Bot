class State:
    def __init__(self, obs, config):
        # complete game configuration - remove eventually
        self.config = config

        # game configuration
        self.map_size = config.size
        self.total_steps = config.episodeSteps
        self.step = obs.step
        self.halite_map = np.array(obs.halite)

        # my halite, yards, and ships
        self.my_id = obs.player
        self.my_halite, self.my_yards, self.my_ships = obs.players[self.my_id]

        # set joint and individual data for oppenents
        self.set_opp_data(obs)

        # several functions want a vector of all sites so we only generate
        # this once and keep it
        size = self.map_size
        nsites = size ** 2
        self.sites = np.arange(nsites)

        # list of positions with ships that have already moved this turn
        self.moved_this_turn = np.zeros_like(self.sites).astype(bool)

        # lookup tables for the effect of moves
        # north[x] is the position north of x, etc
        self.north = (self.sites - size) % nsites
        self.south = (self.sites + size) % nsites
        self.east = size * (self.sites // size) \
            + ((self.sites % size) + 1) % size
        self.west = size * (self.sites // size) \
            + ((self.sites % size) - 1) % size

        # dist[x,y] stores the l1-distance between x and y on the torus
        cols = self.sites % self.map_size
        rows = self.sites // self.map_size
        coldist = cols - cols[:, np.newaxis]
        rowdist = rows - rows[:, np.newaxis]
        coldist = np.fmin(np.abs(coldist), self.map_size - np.abs(coldist))
        rowdist = np.fmin(np.abs(rowdist), self.map_size - np.abs(rowdist))
        self.dist = coldist + rowdist

        # sets a number of numpy arrays deriving from self.my_ships, etc
        self.set_derived()
        return

    def set_opp_data(self, obs):
        # figure out my id / opponent id
        self.opp_ids = list(range(0, len(obs.players)))
        self.opp_ids.remove(self.my_id)

        # joint opponent ships and yards
        self.opp_ships = {}
        self.opp_yards = {}
        for opp in self.opp_ids:
            self.opp_yards.update(obs.players[opp][1])
            self.opp_ships.update(obs.players[opp][2])

        # arrays containing ship/yard data for all opponents
        self.opp_yard_pos = np.array(list(self.opp_yards.values())).astype(int)

        if len(self.opp_ships) != 0:
            poshal = np.array(list(self.opp_ships.values()))
            self.opp_ship_pos, self.opp_ship_hal = np.split(poshal, 2, axis=1)
            self.opp_ship_pos = np.ravel(self.opp_ship_pos).astype(int)
            self.opp_ship_hal = np.ravel(self.opp_ship_hal).astype(int)
        else:
            self.opp_ship_pos = np.array([]).astype(int)
            self.opp_ship_hal = np.array([]).astype(int)

        # now construct a dict of lists with halite, yard positions, ship
        # positions, ship halite for each opponent as numpy arrays
        self.opp_data = {}
        self.opp_scores = {}
        self.opp_num_ships = {}
        for opp in self.opp_ids:
            halite, yards, ships = obs.players[opp]
            yard_pos = np.array(list(yards.values())).astype(int)
            if len(ships) > 0:
                poshal = np.array(list(ships.values()))
                ship_pos, ship_hal = np.split(poshal, 2, axis=1)
                ship_pos = np.ravel(ship_pos).astype(int)
                ship_hal = np.ravel(ship_hal).astype(int)
            else:
                ship_pos = np.array([]).astype(int)
                ship_hal = np.array([]).astype(int)

            self.opp_data[opp] = [halite, yard_pos, ship_pos, ship_hal]
            self.opp_num_ships[opp] = ship_pos.size
            alive = (ship_pos.size + yard_pos.size) > 0
            self.opp_scores[opp] = alive * (halite + np.sum(ship_hal))

        return

    # several function need all our ship/yard positions as numpy arrays
    # these arrays need to be set by init() and also updated by update()
    # do this by calling set_derived()
    def set_derived(self):
        self.my_yard_pos = np.array(list(self.my_yards.values())).astype(int)

        if len(self.my_ships) != 0:
            poshal = np.array(list(self.my_ships.values()))
            self.my_ship_pos, self.my_ship_hal = np.split(poshal, 2, axis=1)
            self.my_ship_pos = np.ravel(self.my_ship_pos).astype(int)
            self.my_ship_hal = np.ravel(self.my_ship_hal).astype(int)
        else:
            self.my_ship_pos = np.array([]).astype(int)
            self.my_ship_hal = np.array([]).astype(int)

        return

    def newpos(self, pos, action):
        if (action is None) or (action == "CONVERT"):
            return pos
        elif action == "NORTH":
            return self.north[pos]
        elif action == "SOUTH":
            return self.south[pos]
        elif action == "EAST":
            return self.east[pos]
        elif action == "WEST":
            return self.west[pos]

    def update(self, actor, action):
        # if actor is a yard only spawning has an effect on state
        if (actor in self.my_yards) and (action == "SPAWN"):
            # create new id string
            newid = f"spawn[{actor}]"

            # create a new ship with no cargo at yard position
            pos = self.my_yards[actor]
            self.my_ships[newid] = [pos, 0]

            # the result is a ship here that cannot move this turn
            self.moved_this_turn[pos] = True

            # subtract spawn cost from available halite
            self.my_halite -= int(self.config.spawnCost)

        if actor in self.my_ships:
            pos, hal = self.my_ships[actor]

            if action == "CONVERT":
                # create a new yard at ship position and remove ship
                self.my_yards[actor] = pos
                del self.my_ships[actor]

                # remove conversion cost from available halite but don't add
                # any net gains - it's not available until next turn
                self.my_halite += min(hal - self.config.convertCost, 0)
                self.halite_map[pos] = 0

            else:
                # if the ship stays put, it can collect halite
                if action is None:
                    collect = self.halite_map[pos] * self.config.collectRate
                    nhal = int(hal + collect)
                    self.halite_map[pos] -= collect
                else:
                    nhal = hal

                # write the new position and halite into my_ships
                npos = self.newpos(pos, action)
                self.my_ships[actor] = [npos, nhal]

                # add new position to list of ships that cannot move this turn
                self.moved_this_turn[npos] = True

        # update internal variables that derive from my_ships, my_yards
        self.set_derived()
        return

    def legal_actions(self, actor):
        if actor in self.my_yards:
            actions = [None, "SPAWN"]
            # need to have enough halite to spawn
            if self.my_halite < self.config.spawnCost:
                actions.remove("SPAWN")

        if actor in self.my_ships:
            actions = [None, "CONVERT", "NORTH", "SOUTH", "EAST", "WEST"]
            pos, hal = self.my_ships[actor]

            # need to have enough halite. if you only have one ship, you
            # can only convert if you still have enough halite afterwards
            # to spawn a new ship
            minhal = self.config.convertCost - hal
            minhal += self.config.spawnCost * (len(self.my_ships) == 1)

            # can't convert if you don't have enough halite or are in a yard
            if (self.my_halite < minhal) or (pos in self.my_yard_pos):
                actions.remove("CONVERT")

        return actions

    def self_collision(self, actor, action):
        if actor in self.my_yards:
            # shipyards only give collisions if we spawn a ship after moving
            # a ship there previously
            pos = self.my_yards[actor]
            collision = (action == "SPAWN") and self.moved_this_turn[pos]

        if actor in self.my_ships:
            pos, hal = self.my_ships[actor]

            # cannot convert onto another shiyard
            if action == "CONVERT":
                collision = pos in self.my_yard_pos
            # otherwise get the new position of the ship and check if
            # it runs into a ship that can no longer move
            else:
                npos = self.newpos(pos, action)
                collision = self.moved_this_turn[npos]

        return collision

    def opp_collision(self, ship, action, strict=True):
        pos = self.my_ships[ship][0]
        npos = self.newpos(pos, action)
        unsafe = self.unsafe_sites(ship, strict)
        return unsafe[npos]

    def unsafe_sites(self, ship, strict=True):
        pos, hal = self.my_ships[ship]

        # can only go to sites with distance <= 1
        unsafe = self.dist[pos, :] > 1

        # find those ships that have less halite than we do
        # add 1 to hal if we want to have a strict halite comparison
        # since x < hal becomes x <= hal for integer values...
        threats = self.opp_ship_pos[self.opp_ship_hal < (hal + strict)]

        # set of sites where ships with less cargo can be in one step
        if threats.size != 0:
            unsafe |= (np.amin(self.dist[threats, :], axis=0) <= 1)

        # opponent yards
        unsafe |= np.in1d(self.sites, self.opp_yard_pos)

        return unsafe
