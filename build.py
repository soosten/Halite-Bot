#!/usr/bin/python

import os
import subprocess
import sys
import tarfile

from kaggle_environments import make


def main():
    # path to repository from working directory
    path = os.curdir

    # can play with 1, 2, or 4 players. None is the idle agent, "random" is
    # the built-in random agent, or one can specify agents by filename:
    # agents = [submission]
    # agents = ["otheragent.py", submission]
    # agents = ["random", submission, "random", None]
    submission = os.path.join(path, "src", "main.py")
    agents = [submission, submission, submission, submission]

    # run the simulation - random seed and number of steps are optional
    run(path, agents, steps=400, seed=13)

    # uncomment to upload submission.py to kaggle competition
    # assumes kaggle CLI is installed with proper credentials
    # submit(path, "#61 - ")

    print("\nDone.")
    return


def run(path, agents, steps=400, seed=None):
    # append files in /src/ to the python path
    src = os.path.join(path, "src")
    sys.path.append(src)

    # get a halite simulator from kaggle environments and run the simulation
    if seed is not None:
        print(f"Running for {steps} steps with seed = {seed}...\n\n")
        config = {"randomSeed": seed, "episodeSteps": steps}
    else:
        print(f"Running for {steps} steps...\n\n")
        config = {"episodeSteps": steps}

    halite = make("halite", debug=True, configuration=config)
    halite.run(agents)

    # render the output video in simulation.html
    print("\nRendering episode...")
    out = halite.render(mode="html", width=800, height=600)
    with open(os.path.join(path, "simulation.html"), "w") as file:
        file.write(out)

    # remove files in /src/ from python path
    sys.path.remove(src)
    return


def submit(path, description):
    # double check whether we want to submit the agent
    if "yes" != input("Are you sure you want to submit? [yes/no] "):
        print("\nNot uploaded.")
        return

    # add all python files in path/src/ to submission.tar.gz
    src = os.path.join(path, "src")
    files = [file for file in os.listdir(src) if file.endswith(".py")]
    submission = os.path.join(path, "submission.tar.gz")
    with tarfile.open(submission, "w:gz") as archive:
        for file in files:
            archive.add(os.path.join(src, file), arcname=file)

    # upload using the kaggle command line tool
    print("")
    cmd = "kaggle competitions submit -q -c halite -f".split()
    cmd.extend((submission, "-m", description))
    subprocess.run(cmd)
    print("")

    # remove the tar archive
    os.remove(submission)
    return


if __name__ == "__main__":
    main()
