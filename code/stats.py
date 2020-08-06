class Stats:
    def __init__(self):
        self.last_state = None
        self.state = None
        self.yard_attackers = []

        self.total_bounties = 0
        self.converted_bounties = 0
        self.loot = 0
        self.yards_destroyed = 0

        self.ships_lost = 0
        self.yards_lost = 0

        self.yards_built = 0
        self.ships_built = 0
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

        # update how many ships/yards we lost
        self.count_actors()

        # save the current state as the previous state
        self.last_state = self.state
        return

    def find_yard_attacks(self):
        # store yard positions for all players for this and last state
        last_all_yard_pos = np.union1d(self.last_state.my_yard_pos,
                                       self.last_state.opp_yard_pos)

        all_yard_pos = np.union1d(self.state.my_yard_pos,
                                  self.state.opp_yard_pos)

        # for every opponent that has not yet attacked a yard, we check
        # whether one of the yards of the other three players has been
        # destroyed during the last turn if one of the opponent ships was
        # next to the destroyed yard on the previous turn, we add that
        # opponent to the list of yard attackers. the check isn't foolproof
        # if there are two opponents near a yard...
        opps = set(self.last_state.opp_data) - set(self.yard_attackers)
        for opp in opps:
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

    def count_actors(self):
        # get ship/yard ids from last step and present step count how many
        # new ones we built and how many were destroyed
        last_yards = set(self.last_state.my_yards)
        yards = set(self.state.my_yards)

        last_ships = set(self.last_state.my_ships)
        ships = set(self.state.my_ships)

        self.yards_lost += len(last_yards - yards)
        self.yards_built += len(yards - last_yards)

        # don't count converted ships as lost
        self.ships_lost += len(last_ships - ships) - len(yards - last_yards)
        self.ships_built += len(ships - last_ships)

        # see if any of the ships we lost destroyed opponent yards
        destroyed = np.setdiff1d(self.last_state.opp_yard_pos,
                                 self.state.opp_yard_pos)

        for ship in last_ships - ships:
            pos = self.last_state.my_ships[ship][0]
            dists = self.state.dist[destroyed, pos]
            self.yards_destroyed += (1 in dists)

        # see if any of the bounties we set was destroyed
        hunted = [self.last_state.opp_ships[key][1] for key in
                  bounties.ship_targets if key not in self.state.opp_ships]
        stats.converted_bounties += len(hunted)
        stats.loot += sum(hunted)

        return

    def summary(self):
        frac = f"{self.converted_bounties}/{self.total_bounties}"
        if self.total_bounties != 0:
            dec = round(self.converted_bounties / self.total_bounties, 2)
            frac = frac + f" = {dec}"

        mined = self.state.my_halite \
              + self.state.config.spawnCost * self.ships_built \
              + self.state.config.convertCost * self.yards_built

        print("")
        print("Bounties converted: " + frac)
        print(f"Total loot: {self.loot}")
        print(f"Yards destroyed: {self.yards_destroyed}")
        print(f"Ships built: {self.ships_built}")
        print(f"Yards built: {self.yards_built}")
        print(f"Ships lost: {self.ships_lost}")
        print(f"Yards lost: {self.yards_lost}")
        print(f"Total mined: {mined}")
        print("")

        return
