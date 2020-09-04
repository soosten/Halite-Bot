# graph weights
MY_WEIGHT = 1
MY_RADIUS = 1
OPP_WEIGHT = 4
OPP_RADIUS = 2
HUNT_WEIGHT = 4
HUNT_RADIUS = 3

# conversion and spawning
YARD_DIST = 6
OPP_YARD_DIST = 5
YARD_RADIUS = 4
MIN_CELLS = 15
YARD_SCHEDULE = [0, 9, 25]
YARD_MAX_STEP = 300
YARD_PROTECTION = True
MIN_SHIPS = 15
SPAWNING_STEP = 250
SPAWNING_OFFSET = 5
STEPS_FINAL = 50  # no conversions or spawns with this many steps left

# bounties and hunting
SHIPS_PER_BOUNTY = 7  # 6 or 7 undecided!
HUNTING_MAX_RATIO = 0.33
HUNTING_STEP = 50
YARD_HUNTING_START = 300
YARD_HUNTING_FINAL = 50
YARD_HUNTING_MIN_SHIPS = 10
YARD_HUNTING_RADIUS = 6

# rate options
STEPS_SPIKE = 25  # steps remaining before spike is added
SPIKE_PREMIUM = 0.5  # spike to deposit everything at the end
RISK_PREMIUM = 0.02  # gets added for each threat within radius RISK_RADIUS
STEPS_INITIAL = 50  # don't add risk before this step
BASELINE_SHIP_RATE = 0.08  # tendency to visit more sites
BASELINE_YARD_RATE = 0.02  # tendency to go to yard / mine for less time
