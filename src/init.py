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
YARD_SCHEDULE = np.array([0, 10, 40])
YARD_MAX_STEP = 300
FIFO_MODE = True
MAX_SHIPS = 70
MIN_SHIPS = 15
SPAWNING_STEP = 250
SPAWNING_OFFSET = 5
STEPS_FINAL = 50  # no conversions or spawns with this many steps left

# bounties and hunting
SHIPS_PER_BOUNTY = 6
HUNTING_MAX_RATIO = 0.33
HUNTING_STEP = 50
YARD_HUNTING_START = 300
YARD_HUNTING_FINAL = 50
YARD_HUNTING_MIN_SHIPS = 10
YARD_HUNTING_RADIUS = 6
MIN_MINING_HALITE = 0

# rate options
STEPS_SPIKE = 30  # steps remaining before spike is added
SPIKE_PREMIUM = 0.8  # spike to deposit everything at the end
RISK_RADIUS = 10
RISK_PREMIUM = 0.02  # gets added for each threat within radius RISK_RADIUS
STEPS_INITIAL = 50  # don't add risk before this step
SPAWN_PREMIUM_STEP = 20  # don't add spawn premium before this step
SPAWN_PREMIUM = 0.01  # deposit if we need halite to spawn
BASELINE_SHIP_RATE = 0.08  # tendency to visit more sites
BASELINE_YARD_RATE = 0.02  # tendency to go to yard / mine for less time

# print how long each step took?
PRINT_STEP_TIMES = False

# intialize global strategy objects we keep throughout the episode
stats = Stats()
bounties = Bounties()
fifos = Fifos()
targets = Targets()
