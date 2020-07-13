#!/usr/bin/python

from os import listdir
from shutil import copyfileobj
from kaggle_environments import make
import sys


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

    # run the simulation - setting a random seed is optional
    run(PATH, agents, seed=42)

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


main()
