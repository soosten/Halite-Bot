from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments.utils import structify
import argparse
import pprint
import json
import sys

# import your submission here
import bug as sub


# replay_match(options.file, options.step, options.id)
def replay_match(path, step, playerid):
  with open(path, 'r') as f:
    match = json.load(f)
  env = make("halite", configuration=match['configuration'], steps=match['steps'])

  state2=match['steps'][step][0]   # list of length 1 for each step
  obs=state2['observation']  # these are observations at this step
  config=env.configuration
  obs['player']=playerid  # change the player to the one we want to inspect
  board=Board(obs,config)
  #check that we are correct player
  print('I am ', board.current_player_id, board.current_player)
  obs=structify(obs)   # turn the dict's into structures with attributes
  #This is our agent recreating what happened on this step
  ret=sub.agent(obs, config)
  print('Returned from agent:',ret)

replay_match('/Users/p/Desktop/1308286.json', 117, 2)
