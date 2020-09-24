# graph weights
MY_WEIGHT = 1  # weight contributed to target graph by friendly ships
MY_RADIUS = 1  # range of influence of each friendly ship on target graph
OPP_WEIGHT = 4  # weight contributed to target graph by opponent ships
OPP_RADIUS = 2  # range of influence of each opponent ship on target graph
HUNT_WEIGHT = 4  # weight contributed to hunting graph by hunting ships
HUNT_RADIUS = 3  # range of influence of each hunting ship on hunting graph

# rate options
BASELINE_SHIP_RATE = 0.08  # tendency to visit more sites
BASELINE_YARD_RATE = 0.02  # tendency to go to yard / mine for less time
SPIKE_PREMIUM = 0.5  # rate spike to deposit everything at the end
STEPS_SPIKE = 25  # steps remaining before spike is added
RISK_PREMIUM = 0.02  # gets added for each threat
STEPS_INITIAL = 50  # don't add risk before this step

# conversion
MIN_YARD_DIST = 6  # minimum distance of new yards to friendly yards
MIN_OPP_YARD_DIST = 6  # minimum distance of new yards to opponent yards
SUPPORT_DIST = 10  # maximum distance of new yards to supporting yards
YARD_SCHEDULE = [0, 11, 18, 35]  # how many ships before we build the next yard
YARD_MAX_STEP = 300  # don't build new yards after this step

# spawning
MIN_SHIPS = 15  # minimum amount of ships to keep
SPAWNING_STEP = 250  # always spawn up to this step
SPAWNING_OFFSET = 5  # spawn if difference to opponent ships exceed this
STEPS_FINAL = 50  # no conversions or spawns with this many steps left

# bounties
SHIPS_PER_BOUNTY = 7  # number of ships required for each new bounty
HUNTING_MAX_RATIO = 0.33  # proportion of starting halite
HUNTING_STEP = 50  # don't hunt earlier than this step
HUNT_NEARBY = 3  # number of nearby hunters required for targeting of ships
