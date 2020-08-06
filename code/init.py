# parameters for graph weighting
GRAPH_MY_WEIGHT = 1
GRAPH_OPP_WEIGHT = 4

# parameters for shipyard conversion
YARD_DIST = 7
YARD_RADIUS = 4
MIN_CELLS = 24
YARD_SCHEDULE = np.array([0, 10, 30])
FIFO_MODE = True

# how many steps are the "initial" and "final" phases of the game
STEPS_INITIAL = 50
STEPS_FINAL = 50

# parameters for spawning decisions
MAX_SHIPS = 70
MIN_SHIPS = 15

# parameters for setting bounties and hunters
SHIPS_PER_BOUNTY = 6
HUNTING_MAX_RATIO = 0.3
YARD_HUNTING_START = 300
YARD_HUNTING_FINAL = 30
YARD_HUNTING_MIN_SHIPS = 10
YARD_HUNTING_RADIUS = 6

# rate options
STEPS_SPIKE = 15
SPIKE_PREMIUM = 0.8
RISK_PREMIUM = 0.02
SPAWN_PREMIUM = 0.03
BASELINE_SHIP_RATE = 0.05
BASELINE_YARD_RATE = 0.01

# intialize global strategy objects we keep throughout the episode
stats = Stats()
bounties = Bounties()
fifos = Fifos()
targets = Targets()
