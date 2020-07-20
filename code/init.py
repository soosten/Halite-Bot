# parameters for graph weighting
GRAPH_MY_WEIGHT = 2
GRAPH_OPP_WEIGHT = 4

# parameters for yard clustering algorithm
MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 5
BASELINE_SHIPS_PER_YARD = 6

# how many steps are the "initial" and "final" phases of the game
STEPS_INITIAL = 50
STEPS_FINAL = 50
STEPS_INTEREST_SPIKE = 15

# parameters for spawning decisions
MAX_SHIPS = 70
MIN_SHIPS = 15

# parameters for setting targets and hunters
HUNTERS_PER_TARGET = 5
MAX_HUNTERS_PER_SHIP = 1
MIN_HUNTERS_PER_SHIP = 0.5

# interest rate
BASELINE_INTEREST = 0  # previously 0.01 - this may be big improvement
SPAWN_PREMIUM = 0

# whether we should operate the shipyards in FIFO mode
USE_FIFO_SYSTEM = True

# intialize global strategy objects we keep throughout the episode
stats = Stats()
targets = Targets()
fifos = Fifos()
