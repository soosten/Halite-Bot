class State:
    def __init__(self, obs, config):
        # read game configuration
        self.map_size = config.size
        self.total_steps = config.episodeSteps
        self.starting_halite = config.startingHalite
        self.regen_rate = config.regenRate
        self.collect_rate = config.collectRate
        self.convert_cost = config.convertCost
        self.spawn_cost = config.spawnCost

        # step and halite map
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
        self.moved_this_turn = np.zeros_like(self.sites, dtype=bool)

        # lookup tables for the effect of moves
        # north[x] is the position north of x, etc
        self.north = (self.sites - size) % nsites
        self.south = (self.sites + size) % nsites
        self.east = size * (self.sites // size) \
            + ((self.sites % size) + 1) % size
        self.west = size * (self.sites // size) \
            + ((self.sites % size) - 1) % size

        # dist[x,y] stores the l1-distance between x and y on the torus
        cols = self.sites % size
        rows = self.sites // size
        coldist = cols - cols[:, np.newaxis]
        rowdist = rows - rows[:, np.newaxis]
        coldist = np.fmin(np.abs(coldist), size - np.abs(coldist))
        rowdist = np.fmin(np.abs(rowdist), size - np.abs(rowdist))
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
        poshal = np.array(list(self.opp_ships.values()), dtype=int)
        pos, hal = np.hsplit(poshal, 2)
        self.opp_ship_pos = np.ravel(pos)
        self.opp_ship_hal = np.ravel(hal)
        self.opp_yard_pos = np.array(list(self.opp_yards.values()), dtype=int)

        # construct a dict of lists with halite, yard positions, ship
        # positions, ship halite for each opponent as numpy arrays
        self.opp_data = {}
        self.opp_scores = {}
        self.opp_num_ships = {}

        for opp in self.opp_ids:
            halite, yards, ships = obs.players[opp]

            poshal = np.array(list(ships.values()), dtype=int)
            ship_pos, ship_hal = np.hsplit(poshal, 2)
            ship_pos = np.ravel(ship_pos)
            ship_hal = np.ravel(ship_hal)
            yard_pos = np.array(list(yards.values()), dtype=int)

            self.opp_data[opp] = [halite, yard_pos, ship_pos, ship_hal]
            self.opp_num_ships[opp] = ship_pos.size
            if ship_pos.size + yard_pos.size > 0:
                self.opp_scores[opp] = halite + np.sum(ship_hal)
            else:
                self.opp_scores[opp] = 0

        return

    # several function need all our ship/yard positions as numpy arrays
    # these arrays need to be set by init() and also updated by update()
    # do this by calling set_derived()
    def set_derived(self):
        poshal = np.array(list(self.my_ships.values()), dtype=int)
        pos, hal = np.hsplit(poshal, 2)
        self.my_ship_pos = np.ravel(pos)
        self.my_ship_hal = np.ravel(hal)
        self.my_yard_pos = np.array(list(self.my_yards.values()), dtype=int)
        return

    def pos_to_move(self, initial, final):
        if final == self.north[initial]:
            return "NORTH"
        elif final == self.south[initial]:
            return "SOUTH"
        elif final == self.east[initial]:
            return "EAST"
        elif final == self.west[initial]:
            return "WEST"
        else:
            return None

    def move_to_pos(self, initial, move):
        if move == "NORTH":
            return self.north[initial]
        elif move == "SOUTH":
            return self.south[initial]
        elif move == "EAST":
            return self.east[initial]
        elif move == "WEST":
            return self.west[initial]
        else:
            return initial

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
            self.my_halite -= int(self.spawn_cost)

        if actor in self.my_ships:
            pos, hal = self.my_ships[actor]

            if action == "CONVERT":
                # create a new yard at ship position and remove ship
                self.my_yards[actor] = pos
                del self.my_ships[actor]

                # remove conversion cost from available halite but don't add
                # any net gains - it's not available until next turn
                self.my_halite += min(hal - self.convert_cost, 0)
                self.halite_map[pos] = 0

            else:
                # if the ship stays put, it can collect halite
                if action is None:
                    collect = self.halite_map[pos] * self.collect_rate
                    nhal = int(hal + collect)
                    self.halite_map[pos] -= collect
                else:
                    nhal = hal

                # write the new position and halite into my_ships
                npos = self.move_to_pos(pos, action)
                self.my_ships[actor] = [npos, nhal]

                # add new position to list of ships that cannot move this turn
                self.moved_this_turn[npos] = True

        # update internal variables that derive from my_ships, my_yards
        self.set_derived()
        return
