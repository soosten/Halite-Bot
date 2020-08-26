#!/usr/bin/python

import os
import subprocess
from shutil import copyfileobj
from kaggle_environments import make


def main():
    # path to repository from working directory
    path = os.curdir

    # write the source files into a file submission.py with the right format
    write(path)
    submission = os.path.join(path, "submission.py")

    # can play with 1, 2, or 4 players. None is the idle agent, "random" is
    # the built-in random agent, or one can specify agents by filename:
    # agents = [submission]
    # agents = ["otheragent.py", submission]
    # agents = ["random", submission, "random", None]
    agents = [submission, submission, submission, submission]

    # run the simulation - random seed and number of steps are optional
    run(path, agents, steps=400, seed=13)

    # uncomment to upload submission.py to kaggle competition
    # assumes kaggle CLI is installed with proper credentials
    # submit(path, "#45 - ")

    print("\nDone.")
    return


def write(path):
    # get files in path/src/ that are not system files
    src_path = os.path.join(path, "src")
    files = [name for name in os.listdir(src_path) if name.endswith(".py")]

    # check if agent.py, init.py, and imports.py exist and remove them
    try:
        files.remove("agent.py")
        files.remove("init.py")
        files.remove("imports.py")
    except ValueError:
        print("Error: /src/ directory must contain agent.py, "
              + "imports.py, and init.py")
        raise SystemExit

    # write the files in lexicographical order so its easier to
    # scroll to them in the combined file
    files.sort()

    # write imports.py, then all files in path/src/, then init.py,
    # and finally agent.py into submission.py
    print("Writing files...")
    with open(os.path.join(path, "submission.py"), "w") as sub_file:
        print("  imports.py")
        with open(os.path.join(src_path, "imports.py"), "r") as file:
            copyfileobj(file, sub_file)
        sub_file.write("\n\n")

        for name in files:
            print("  " + name)
            with open(os.path.join(src_path, name), "r") as file:
                copyfileobj(file, sub_file)
            sub_file.write("\n\n")

        print("  init.py")
        with open(os.path.join(src_path, "init.py"), "r") as file:
            copyfileobj(file, sub_file)
        sub_file.write("\n\n")

        print("  agent.py")
        with open(os.path.join(src_path, "agent.py"), "r") as file:
            copyfileobj(file, sub_file)

    return


def run(path, agents, steps=400, seed=None):
    # get a halite simulator from kaggle environments
    if seed is not None:
        print(f"\nRunning for {steps} steps with seed = {seed}...\n\n")
        env = make("halite", debug=True,
                   configuration={"randomSeed": seed, "episodeSteps": steps})
    else:
        print(f"\nRunning for {steps} steps...\n\n")
        env = make("halite", debug=True, configuration={"episodeSteps": steps})

    # run the simulation
    env.run(agents)

    # write the output video to simulation.html
    print("\nRendering episode...")
    out = env.render(mode="html", width=800, height=600)
    simulation = os.path.join(path, "simulation.html")
    with open(simulation, "w") as file:
        file.write(out)

    return


def submit(path, description):
    # double check whether we want to submit the agent
    if "yes" != input("Are you sure you want to submit? [yes/no] "):
        print("\nNot uploaded.")
        return

    # if yes upload using the kaggle command line tool
    print("")
    submission = os.path.join(path, "submission.py")
    cmd = "kaggle competitions submit -q -c halite -f".split()
    cmd.extend((submission, "-m", description))
    subprocess.run(cmd)
    print("")
    return


if __name__ == "__main__":
    main()
