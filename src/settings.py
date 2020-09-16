# graph weights
MY_WEIGHT = 1
MY_RADIUS = 1
OPP_WEIGHT = 4
OPP_RADIUS = 2
HUNT_WEIGHT = 4
HUNT_RADIUS = 3

# targeting
STEPS_SPIKE = 25  # steps remaining before spike is added
SPIKE_PREMIUM = 0.5  # spike to deposit everything at the end
STEPS_INITIAL = 50  # don't add risk before this step
BASELINE_SHIP_RATE = 0.08  # tendency to visit more sites
BASELINE_YARD_RATE = 0.02  # tendency to go to yard / mine for less time
RISK_PREMIUM = 0.02  # gets added for each threat

# conversion
MIN_YARD_DIST = 6
MIN_OPP_YARD_DIST = 6
SUPPORT_DIST = 9
YARD_SCHEDULE = [0, 11, 25, 40, 55]
YARD_MAX_STEP = 300

# spawning
MIN_SHIPS = 15
SPAWNING_STEP = 250
SPAWNING_OFFSET = 5
STEPS_FINAL = 50  # no conversions or spawns with this many steps left

# bounties
SHIPS_PER_BOUNTY = 7
HUNTING_MAX_RATIO = 0.33
HUNTING_STEP = 50
HUNT_NEARBY = 2  # prelim, maybe 1 or 2?
