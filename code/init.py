# parameters for graph weighting
GRAPH_MY_WEIGHT = 1
GRAPH_OPP_WEIGHT = 4

# parameters for shipyard conversion
YARD_DIST = 7
YARD_RADIUS = 4
MIN_CELLS = 20
YARD_SCHEDULE = np.array([0, 0, 30])  # second 0?
FIFO_MODE = True

# how many steps are the "initial" and "final" phases of the game
STEPS_INITIAL = 50
STEPS_FINAL = 50

# parameters for spawning decisions
MAX_SHIPS = 70
MIN_SHIPS = 15

SPAWN_BUFFER = 0 # SET
SPAWN_OFFSET = 0

# parameters for setting bounties and hunters
SHIPS_PER_BOUNTY = 5
HUNTING_MAX_RATIO = 0.3
YARD_HUNTING_START = 330
YARD_HUNTING_FINAL = 30
YARD_HUNTING_MIN_SHIPS = 10
YARD_HUNTING_RADIUS = 6

# rate options
STEPS_SPIKE = 15  # steps remaining before spike is added
SPIKE_PREMIUM = 0.8  # spike to deposit everything at the end
RISK_PREMIUM = 0.02  # gets added for each threat within radius 10
SPAWN_PREMIUM = 0.01  # deposit if we need halite to spawn
BASELINE_SHIP_RATE = 0.08  # tendency to visit more sites
BASELINE_YARD_RATE = 0.02  # tendency to go to yard / mine for less time

# intialize global strategy objects we keep throughout the episode
stats = Stats()
bounties = Bounties()
fifos = Fifos()
targets = Targets()
