#!/usr/bin/python

from os import listdir
from shutil import copyfileobj
from kaggle_environments import make
from random import randint


def main():
    # path to repository from working directory
    PATH = "./"

    # write the source files into a file submission.py with the right format
    write(PATH)

    # can play with 1, 2, or 4 players. None is the idle agent, "random" is
    # the built-in random agent, or one can specify agents by filename, i.e.
    # agents = [PATH + "submission.py"]
    # agents = ["./otheragent.py", PATH + "submission.py"]
    # agents = ["random", PATH + "submission.py", "random", None]
    # this is the "validation episode" run by kaggle:
    agents = [PATH + "submission.py", PATH + "submission.py",
              PATH + "submission.py", PATH + "submission.py"]

    # run the simulation - random seed and number of steps are optional
    # here we keep the seed random but set to a definite value so that
    # the outcomes of individual episodes can be reproduced
    run(PATH, agents, steps=400, seed=randint(1, 1000))

    print("\nDone.")
    return


def write(PATH):
    # list of files in PATH/src/ that are not system files
    src_dir = [name for name in listdir(PATH + "src/") if name[0] != "."]

    # check if agent.py, init.py, and imports.py exist and remove them
    try:
        src_dir.remove("agent.py")
        src_dir.remove("init.py")
        src_dir.remove("imports.py")

    except ValueError:
        print("Error: /src/ directory must contain agent.py, "
              + "imports.py, and init.py")
        raise SystemExit

    # write the files in lexicographical order so its easier to
    # scroll to them in the combined file
    src_dir.sort()

    # write imports.py, then all files in PATH/src/, then init.py,
    # and finally agent.py into submission.py
    print("Writing files...")
    with open(PATH + "submission.py", "w") as submission:
        print("  imports.py")
        with open(PATH + "src/imports.py", "r") as file:
            copyfileobj(file, submission)
        submission.write("\n\n")

        for name in src_dir:
            print("  " + name)
            with open(PATH + "src/" + name, "r") as file:
                copyfileobj(file, submission)
            submission.write("\n\n")

        print("  init.py")
        with open(PATH + "src/init.py", "r") as file:
            copyfileobj(file, submission)
            submission.write("\n\n")

        print("  agent.py")
        with open(PATH + "src/agent.py", "r") as file:
            copyfileobj(file, submission)

    return


def run(PATH, agents, seed=None):
    # get a halite simulator from kaggle environments
    if seed is not None:
        print(f"\nRunning simulation with seed = {seed}...\n\n")
        env = make("halite", debug=True, configuration={"randomSeed": seed})
    else:
        print("\nRunning simulation...\n\n")
        env = make("halite", debug=True)

    # run the simulation
    env.run(agents)

    # write the output video to simulation.html
    print("\nRendering episode...")
    out = env.render(mode="html", width=800, height=600)
    with open(PATH + "simulation.html", "w") as file:
        file.write(out)

    return


if __name__ == "__main__":
    main()
