#!/usr/bin/python

from os import listdir
from shutil import copyfileobj
from kaggle_environments import make
import sys
import subprocess
import random


def main():
    # path to repository from working directory
    PATH = "./"
    write(PATH)

    agents = [PATH + "submission.py", PATH + "submission.py",
              PATH + "submission.py", PATH + "submission.py"]

    # agents = ["random", "./submission.py", "./8.py", "./oernie.py"]
    # agents = ["./oernie.py", "./submission.py", "./oernie.py", "./oernie.py"]
    agents = ["./oernie.py", "./submission.py", "./no1.py", "./8.py"]
    # agents = ["random", "submission.py", "random", "random"]
    run(PATH, agents)

    # description = "#20 - targeting updates"
    # submit(PATH, description)

    print("Done.")
    return


def write(PATH):
    # list of files in PATH/code/ that are not system files
    code_dir = [name for name in listdir(PATH + "code/") if name[0] != "."]

    # check if agent.py, init.py, and imports.py exist and remove them
    try:
        code_dir.remove("agent.py")
        code_dir.remove("init.py")
        code_dir.remove("imports.py")

    except ValueError:
        print("Error: /code/ directory must contain agent.py, "
              + "imports.py, and init.py")
        raise SystemExit

    # write imports.py, then all files in PATH/code/, then init.py,
    # and finally agent.py into submission.py
    print("Writing files...")
    with open(PATH + "submission.py", "w") as submission:
        print("  imports.py")
        submission.write("# ------------ CODE IMPORTED FROM "
                         + "imports.py ------------ #\n")
        with open(PATH + "code/imports.py", "r") as file:
            copyfileobj(file, submission)
        submission.write("\n\n")

        for name in code_dir:
            print("  " + name)
            submission.write("# ------------ CODE IMPORTED FROM "
                             + name + " ------------ #\n")
            with open(PATH + "code/" + name, "r") as file:
                copyfileobj(file, submission)
            submission.write("\n\n")

        print("  init.py")
        submission.write("# ------------ CODE IMPORTED FROM "
                         + "init.py ------------ #\n")
        with open(PATH + "code/init.py", "r") as file:
            copyfileobj(file, submission)
            submission.write("\n\n")

        print("  agent.py")
        submission.write("# ------------ CODE IMPORTED FROM "
                         + "agent.py ------------ #\n")
        with open(PATH + "code/agent.py", "r") as file:
            copyfileobj(file, submission)

    return


def run(PATH, agents, seed=None):
    # get a halite simulator from kaggle environments
    if seed is not None:
        print(f"Running simulation with seed = {seed}...")
        env = make("halite", debug=True, configuration={"randomSeed": seed})
    else:
        print("Running simulation...")
        env = make("halite", debug=True)

    # redirect stdout/stderr to console.txt and run the simulation
    realstdout = sys.stdout
    realstderr = sys.stderr

    with open(PATH + "console.txt", "w") as sys.stdout:
        sys.stderr = sys.stdout
        env.run(agents)

    # restore stdout/stderr
    sys.stdout = realstdout
    sys.stderr = realstderr

    # write the output video to simulation.html
    print("Rendering episode...")
    out = env.render(mode="html", width=800, height=600)
    with open(PATH + "simulation.html", "w") as file:
        file.write(out)

    return


def submit(PATH, description):
    # double check whether we want to submit the agent
    sure = input("Are you sure you want to submit? [yes/no] ")

    # if yes upload using the kaggle command line tool
    if sure == "yes":
        print("\nUploading " + PATH + "submission.py...")
        cmd = "kaggle competitions submit -q -c halite -f " \
            + PATH + "submission.py -m"
        args = cmd.split()
        args.append(description)
        subprocess.run(args)
        print("")
    else:
        print("\nNot uploaded.")

    return


# some primitive agents - mainly useful for debugging
# this one converts on first step and then does nothing
def single_yard(obs, config):
    if obs.step == 0:
        ship = list(obs.players[obs.player][2].keys())[0]
        return {ship: "CONVERT"}
    else:
        return {}


# this one makes random moves with one ship
def single_ship(obs, config):
    ship = list(obs.players[obs.player][2].keys())[0]
    if random.randint(1, 10) == 5:
        return {}
    move = random.choice(["NORTH", "SOUTH", "EAST", "WEST"])
    return {ship: move}


main()
