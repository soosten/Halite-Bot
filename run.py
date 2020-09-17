#!/usr/bin/python

import os
import sys

from kaggle_environments import make


def main():
    # insert path to repository from working directory here
    path = os.curdir
    bot = os.path.join(path, "src", "main.py")

    # can play with 1, 2, or 4 players. None is the idle agent, "random" is
    # the built-in random agent, or one can specify agents by filename:
    # agents = [bot]
    # agents = ["otheragent.py", bot]
    # agents = ["random", bot, "random", None]
    agents = ["random", bot, bot, "random"]

    # number of steps to run the simulation for - the bot's strategy
    # only makes sense if this is 400, which was its value during the
    # competition on kaggle.com
    steps = 400

    # random seed for kaggle environment - set to None to omit
    # seed = None
    seed = 13

    # append files in /src/ to the python path
    src = os.path.join(path, "src")
    sys.path.append(src)

    # get a halite simulator from kaggle environments and run the simulation
    if seed is not None:
        print(f"Running for {steps} steps with seed = {seed}...")
        config = {"randomSeed": seed, "episodeSteps": steps}
    else:
        print(f"Running for {steps} steps...")
        config = {"episodeSteps": steps}

    halite = make("halite", debug=True, configuration=config)
    halite.run(agents)

    # render the output video in simulation.html
    print("Rendering episode...")
    out = halite.render(mode="html", width=800, height=600)
    with open(os.path.join(path, "simulation.html"), "w") as file:
        file.write(out)

    # remove files in /src/ from python path
    sys.path.remove(src)
    print("Done.")
    return


if __name__ == "__main__":
    main()
