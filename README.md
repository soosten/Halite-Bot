# Halite Bot

### About
This repository contains the source code of a bot for the fourth season of [Halite](https://halite.io/) by [Two Sigma](https://www.twosigma.com/), which was hosted on [Kaggle](https://www.kaggle.com/c/halite). The bot was ranked ~10th out of 1143 teams on the [final leaderboard](https://www.kaggle.com/c/halite/leaderboard).


### Local simulations
The script [run.py](run.py) lets the bot play locally against other user-supplied bots or built-in agents and produces a video rendering of the resulting episode in `simulation.html`. This requires the `kaggle_environments` package, which can be installed with the command
```
pip install kaggle_environments
```


### Main components of the strategy
The rules of the game can be found [here](https://www.kaggle.com/c/halite/overview/halite-rules) - the TLDR version is that each player aims to use their ships to bring as much halite as possible home to their shipyards and that in case of ship collisions only the ship with the least cargo survives and takes over the cargo of the other colliding ships.

The general strategy of this bot is to build a large fleet of general-purpose ships that can adapt to the rapidly changing game situations of each match. The fleet tries to expand the space it controls by hunting down vulnerable opponent ships and finances new ships by mining the halite in safe areas close to its own shipyards. [Here](https://www.kaggle.com/c/halite/leaderboard?dialog=episodes-episode-3356885) is an example (we are purple) where this worked out well, whereas [this](https://www.kaggle.com/c/halite/leaderboard?dialog=episodes-episode-3244734) example (we are green) shows the limitations of the strategy.

During each turn, the bot performs the following five steps:

**1. Convert ships into shipyards** <br>
We try to keep a predetermined number of shipyards based on how many ships we have. To place them, we simply check that they are far enough from both opponent shipyards and our own shipyards. If we need a new shipyard and several ships are at locations satisfying these criteria, we convert the ship with the most halite cells nearby. This is the first step, so that the decisions made for other ships in Steps 2-5 can take into account the locations of any newly constructed shipyards.

**2. Place bounties on opponent ships and shipyards** <br>
Throughout each episode we keep a list of opponent ships that we want to hunt down. We add ships to this list if we have enough free ships to go after further targets and remove ships from the list if they escape by getting too close to their own shipyards. The goal is to attack those ships that are most vulnerable. Clearly, one component of vulnerability is the amount of cargo the ship is carrying. The other component we consider is the distance of the ship to the nearest friendly shipyard on a weighted graph. In this weighted graph, each of our likely attacking ships contributes to the weight in a small neighborhood of its position. If this weighted distance is much larger than the distance on the unweighted graph, we take this as an indicator that our attacking ships are blocking the possible escape paths of the target.

For each ship on the list, we "add bounties" to that ships position and the four adjacent sites, which play a role in Step 3.

In the final stages of the episode we also place bounties on the shipyards of the opponent with the closest score to our own, provided we have more ships than everyone else. The main purpose of this is to force this opponent to spend their stored halite on ships and shipyards. This is essentially an attempt at transforming our ships back into stored halite at the end of the game, by forcing a net negative effect on the opponent's halite store.

**3. Determine destinations for ships** <br>
We assign ships to destinations in a such a way that the sum of the rewards is maximized. Up to some details, the reward of a ship going to a site takes the form

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\large \displaystyle R = \frac{B}{(1 %2B r_H)^{d_S}} %2B \max_{m=1,2,\dots} \frac{C %2B \alpha \, (1-\beta^m) H}{(1 %2B r_S)^{d_S} \, (1 %2B r_Y)^{d_Y %2B m}}.">
</p>

In this formula, the numerators are supposed to represent the potential profit available at the site, while the denominators are discount factors that exponentially penalize the risk/effort of going to the site and bringing the cargo home to the nearest shipyard. In the numerators, <img src="https://render.githubusercontent.com/render/math?math=B"> is the value of any bounties placed on this site in Step 2, <img src="https://render.githubusercontent.com/render/math?math=C"> is the amount of cargo carried by the ship, and  <img src="https://render.githubusercontent.com/render/math?math=H"> is the amount of halite available at the site. The numbers <img src="https://render.githubusercontent.com/render/math?math=\alpha"> and <img src="https://render.githubusercontent.com/render/math?math=\beta"> are chosen such that the quantity

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\large \displaystyle \alpha \, (1-\beta^m) H">
</p>

equals the amount of halite the ship picks up if, <img src="https://render.githubusercontent.com/render/math?math=d_S"> steps into the future, it mines the site for <img src="https://render.githubusercontent.com/render/math?math=m"> steps (a simple calculation involving geometric series).

The discounting factors take into account the distance <img src="https://render.githubusercontent.com/render/math?math=d_S"> from the ship's current position to the site and the distance <img src="https://render.githubusercontent.com/render/math?math=d_Y"> from the site to the nearest shipyard. These distances are computed on a weighted graph, where each obstacle contributes to the weights in a neighborhood of its position. The obstacles can be opponent shipyards, opponent ships with less cargo, and also our own ships (which we should not run into). Finally, the rates <img src="https://render.githubusercontent.com/render/math?math=r_S">, <img src="https://render.githubusercontent.com/render/math?math=r_Y">, and <img src="https://render.githubusercontent.com/render/math?math=r_H">  depend on the game situation - we add a (very small) risk premium for each opponent ship that can attack us.

The reason for maximizing over <img src="https://render.githubusercontent.com/render/math?math=m"> is that ships can choose for how long they want to mine each site. Putting the maximum in the formula made it more consistent from step to step and made it less likely that ships decide on a new destination halfway en route to their original destinations. Although it served a different purpose there, the idea of maximizing over <img src="https://render.githubusercontent.com/render/math?math=m"> came from [this](https://www.kaggle.com/solverworld/optimal-mining) beautiful notebook.


**4. Move ships towards their destinations** <br>
This step is very similar to Step 3. Now that we have a destination for each ship, we give scores to the squares adjacent to a ship's current position depending on how much closer they are to the ship's destination on the weighted graphs from Step 3. We also put a large penalty on any squares that are unsafe because they are controlled by opponent ships with less cargo. In order to break ties, we weight the move scores of each ship by a multiplier which depends on the amount of cargo and the distance to the destination of each ship.

Then, like in Step 3, we assign moves to ships in such a way that the sum of scores is maximized. This approach has the nice side effect that collisions between our own ships are avoided automatically. Finally, we double check the result of the optimization problem for any moves that result in a ship being assigned to an unsafe square and try to resolve these cases in a rule-based way.

**5. Spawn new ships** <br>
In the last step, we spawn new ships if there is a lot of time left in the game, if we have fewer ships than one of our opponents, or if we have a lot more stored halite than our opponents. However, near the end of the game we spawn only if our fleet sinks below a certain minimum number of ships.

Finally, there are obviously a number of details (how to keep opponents from running over our shipyards, etc.). The code has a lot comments explaining these things in case anyone should actually be interested in them.


### Overview of the code
The high-level logic of the bot is implemented in [main.py](src/main.py), which is the file compiled by the Kaggle environment. At each step, the function `agent()` is called and passed the current game state in `(obs, config)`. It is expected to return a dictionary whose keys are our ships and shipyards and whose values are the commands issued to those ships and shipyards.

The remaining structure of the code is fairly self-explanatory. The files [convert.py](src/convert.py), [bounties.py](src/bounties.py), [targets.py](src/targets.py), [move.py](src/move.py), and [spawns.py](src/spawns.py) contain the implementations of Steps 1-5 above and are used in order by `agent()`. The file [state.py](src/state.py) contains code that parses the game state provided by the Kaggle environment into a more convenient format and computes a number of derived quantities. Finally, [settings.py](src/settings.py) is a collection of parameters controlling the precise behavior of the bot in Steps 1-5.
