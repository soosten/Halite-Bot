# parameters for graph weighting
GRAPH_MY_WEIGHT = 2
GRAPH_OPP_WEIGHT = 4

# parameters for shipyard conversion
MIN_YARD_DIST = 7
HALITE_RADIUS = 4
MIN_HALITE_CELLS = 20
BASELINE_SHIPS_PER_YARD = 8

# how many steps are the "initial" and "final" phases of the game
STEPS_INITIAL = 50
STEPS_FINAL = 50
STEPS_INTEREST_SPIKE = 15

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

# rate option
BASELINE_INTEREST = 0.01
SPAWN_PREMIUM = 0.01
STEPS_SPIKE = 15

# whether we should operate the shipyards in FIFO mode
USE_FIFO_SYSTEM = True

# intialize global strategy objects we keep throughout the episode
stats = Stats()
bounties = Bounties()
fifos = Fifos()
conversions = Conversions()
targets = Targets()
