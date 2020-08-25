class Memory:
    def __init__(self):
        self.last_state = None
        self.state = None

        self.total_bounties = 0
        self.converted_bounties = 0
        self.loot = 0
        self.yards_destroyed = 0

        self.ships_lost = 0
        self.yards_lost = 0

        self.yards_built = 0
        self.ships_built = 0

        self.total_time = 0

        self.ship_targets = []
        return

    def statistics(self, argstate):
        # on the first turn, just copy the state into last_state and return
        if self.last_state is None:
            self.last_state = deepcopy(argstate)
            return

        # use deepcopy so we keep the state at the beginning of our turn
        # and don't update as we go through deciding actions for our actors
        self.state = deepcopy(argstate)

        # get ship/yard ids from last step and present step count how many
        # new ones we built and how many were destroyed
        last_yards = set(self.last_state.my_yards)
        yards = set(self.state.my_yards)

        last_ships = set(self.last_state.my_ships)
        ships = set(self.state.my_ships)

        self.yards_lost += len(last_yards - yards)
        self.yards_built += len(yards - last_yards)

        # don't count converted ships as lost and don't count any losses
        # after the interest rate spike
        self.ships_built += len(ships - last_ships)
        if self.state.total_steps - self.state.step > STEPS_SPIKE:
            self.ships_lost += len(last_ships - ships)
            self.ships_lost -= len(yards - last_yards)

        # see if any of the ships we lost destroyed opponent yards
        # and don't count these as ships lost
        destroyed = np.setdiff1d(self.last_state.opp_yard_pos,
                                 self.state.opp_yard_pos)

        for ship in last_ships - ships:
            pos = self.last_state.my_ships[ship][0]
            dists = self.state.dist[destroyed, pos]
            self.yards_destroyed += (1 in dists)
            if self.state.total_steps - self.state.step > STEPS_SPIKE:
                self.ships_lost -= (1 in dists)

        # see if any of the bounties we set was destroyed
        hunted = [self.last_state.opp_ships[key][1] for key in
                  self.ship_targets if key not in self.state.opp_ships]
        self.converted_bounties += len(hunted)
        self.loot += sum(hunted)

        # save the current state as the previous state
        self.last_state = self.state
        return

    def summary(self):
        frac = f"{self.converted_bounties}/{self.total_bounties}"
        if self.total_bounties != 0:
            dec = round(self.converted_bounties / self.total_bounties, 2)
            frac = frac + f" = {dec}"

        total = self.state.my_halite \
              + self.state.config.spawnCost * self.ships_built \
              + self.state.config.convertCost * self.yards_built

        avg_time = round(self.total_time / (self.state.total_steps - 1), 2)

        print(f"SUMMARY FOR PLAYER {self.state.my_id}:")
        print("  Bounties converted: " + frac)
        print(f"  Total loot: {self.loot}")
        print(f"  Yards destroyed: {self.yards_destroyed}")
        print(f"  Ships built: {self.ships_built}")
        print(f"  Yards built: {self.yards_built}")
        print(f"  Ships lost: {self.ships_lost}")
        print(f"  Yards lost: {self.yards_lost}")
        print(f"  Total halite: {total}")
        print(f"  Average time per step: {avg_time} seconds\n")
        return
