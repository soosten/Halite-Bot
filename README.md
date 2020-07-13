# Halite
This repository contains the code of an agent for the fourth season of Halite by Two Sigma, which was hosted on [kaggle](https://www.kaggle.com/c/halite). The agent finished Xth out of X teams on the [final leaderboard](https://www.kaggle.com/c/halite).

### Format
An entry to this competition consists of a file `submission.py`, which is processed by kaggle in the following way:
1. The very last function in the file is called at every turn. It is passed the game state in `(obs, config)` and is expected to return a legal list of actions to take for that turn.
2. Any code before the last function executes only once at the beginning of the episode. It can be used to define various classes/functions and also to initialize any global variables to be kept throughout the episode.

### Dependencies
The code depends only on standard libraries (mainly numpy, scipy, and sklearn). However, the script `build.py` requires the `kaggle_environments` package to run and render the simulations locally. The package can be installed with
`pip install kaggle_environments`

### Build
The script `build.py` has code to build a valid submission and run it. The function `write()` processes the code in `/code/` into a file `submission.py` satisfying the criteria above. The function `run()` can be used to run an episode locally, in which the agent can play against other user-supplied agents or the built-in random/idle agents. The output consists of `console.txt`, which contains anything printed to stdout/stderr during the simmulation, and `simulation.html`, which contains a video rendering of the episode.

### Main components of the strategy

### Overview of the code
