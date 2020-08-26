# Halite Bot

### About
This repository contains the source code of a bot for the fourth season of [Halite](halite.io/) by [Two Sigma](www.twosigma.com/), which was hosted on [Kaggle](www.kaggle.com/c/halite). The bot was ranked Xth out of X teams on the [final leaderboard](www.kaggle.com/c/halite/leaderboard).


### Dependencies
The bot depends only on standard libraries and the SciPy stack. However, the script `build.py` requires the `kaggle_environments` package to run and render the simulations locally. It also requires the Kaggle CLI and valid credentials to upload the submission to the Kaggle competition (see [here](github.com/Kaggle/kaggle-api)). These requirements can be installed with

`pip install kaggle`

`pip install kaggle_environments`


### Build
The script `build.py` has functions to build a valid submission, play locally against other user-supplied bots or built-in agents, and upload the submission to the Kaggle competition. It creates a file `simulation.html`, which contains a video rendering of the episode.


### Main components of the strategy
todo
[Dijkstra's algorithm](en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
[Hungarian algorithm](en.wikipedia.org/wiki/Hungarian_algorithm)

### Overview of the code
An entry to the competition consists of a file `submission.py`, which is processed by the Kaggle environment in the following way:
1. The very last function in the file is called at every turn. It is passed the game state in `(obs, config)` and is expected to return a legal list of actions to take for that turn.
2. Any code before the last function executes only once at the beginning of the episode. It can be used to define various classes/functions and to initialize any global variables to be kept throughout the episode.

todo
