from animalai.envs.arena_config import Vector3, RGB, Item, Arena, ArenaConfig
from animalai.envs.environment import AnimalAIEnvironment
from mlagents_envs.exception import UnityCommunicationException

from typing import List
from animalai.communicator_objects import (
    ArenasConfigurationsProto,
    ArenaConfigurationProto,
    ItemToSpawnProto,
    VectorProto,
)

from typing import List
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import networkx as nx
from networkx import grid_graph

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from mlagents.trainers.trainer_util import load_config;
from animalai_train.run_options_aai import RunOptionsAAI;
from animalai_train.run_training_aai import run_training_aai;

import os
import pandas as pd

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

#-----------------------------go from matrix of state to AnimalAI environment------------------------------

class MyArenaConfig(yaml.YAMLObject):
    '''It reads an arena object and create an arena configuration file.
    As ArenaConfig (animalai library), but instead of reading a .yaml file reads an arena object'''
    
    yaml_tag = u"!ArenaConfig"

    def __init__(self, my_arena: Arena = None):

        self.arenas = {-1: my_arena}

    def to_proto(self, seed: int = -1) -> ArenasConfigurationsProto:
        arenas_configurations_proto = ArenasConfigurationsProto()
        arenas_configurations_proto.seed = seed

        for k in self.arenas:
            arenas_configurations_proto.arenas[k].CopyFrom(self.arenas[k].to_proto())

        return arenas_configurations_proto
    
def matrix2arena(matrix):
    '''Given an input numpy matrix of the arena image, it create an AnimalAI environment configuration.
    Matrix values --> 0: arena, 1: goal, 2: agent, 3: wall'''
    if matrix.shape != (40, 40):
        print('Input matrix should have shape 40x40')
        return 
    
    items = []
    
    # create goal item
    position = np.where(matrix==1)
    position_goal = Vector3(x = position[0][0], y = 0, z = position[1][0])
    sizes_goal = Vector3(x = 2, y = 2, z = 2)
    goal = Item(name = 'GoodGoal', positions = [position_goal], sizes = [sizes_goal])
    items.append(goal)

    # create agent item
    position = np.where(matrix==2)
    position_agent = Vector3(x = position[0][0], y = 0, z = position[1][0]) 
    agent = Item(name = 'Agent', positions = [position_agent], rotations = [0]) # we also fix the rotation
    items.append(agent)

    # create walls
    position = np.where(matrix==3)
    for i in range(len(position[0])):
        position_wall= Vector3(x = position[0][i], y = 0, z = position[1][i])
        sizes_wall = Vector3(x = 2, y = 2, z = 2) # fixed size of the wall
        color_wall = RGB(r = 200, g = 0, b = 4) # make wall red
        wall = Item(name = 'Wall', positions = [position_wall], sizes = [sizes_wall], colors = [color_wall], rotations = [0])
        items.append(wall)

    # Create Arena
    my_arena = Arena(t=250, items=items, pass_mark = 0, blackouts = None)
    my_config = MyArenaConfig(my_arena)
        
    
    return my_config

def visualize(state, show=True, save=False, n_episode=[]):
    '''Plots a given state matrix'''
    fig = plt.imshow(state, cmap=plt.get_cmap('Accent'))
    # values
    values = np.array([0, 1, 2, 3])
    #items
    items = ['Arena', 'Goal', 'Agent', 'Wall']
    # colormap used by imshow
    colors = [fig.cmap(fig.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=items[i]) ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.grid(False)
    
    if show:
        plt.show()
        
    if save:
        plt.savefig('figures/env' + str(n_episode) + '.png')
        np.save('matrix/env' + str(n_episode) + '.npy', state)
        
    
#---------------------------------------Check working environment------------------------
    
def run_one_episode(env, verbose=False):
    '''Given a gym environment (env) it computer one episode and return cumulative reward.
    Used to check if our gym environments work'''
    env.reset()
    sum_reward = 0

    for i in range(env.MAX_STEPS+1):
        action = env.action_space.sample()
        if verbose:
            print("action:", action)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            env.render()

        if done:
            if verbose:
                print("done @ step {}".format(i))
            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward
    
#----------------------------------Train agent----------------------------------------------

## RESET / MODIFY CONFIGURATIONS

def reset_trainer(trainer_config_path, steps=1.0e4):
    '''Reset the trainer to a certain number of max steps
    Args:
        trainer_config_path: Path to the .yml file with the trainer configurantion.
        steps: nb of max steps
    '''
    with open(trainer_config_path) as f:
         list_doc = yaml.load(f)

    list_doc['AnimalAI']['max_steps'] = steps

    with open(trainer_config_path, "w") as f:
        yaml.dump(list_doc, f)
        
def update_trainer(trainer_config_path, steps_add=1.0e4):
    '''Reset the trainer to a certain number of max steps
    Args:
        trainer_config_path: Path to the .yml file with the trainer configurantion.
        steps_add: nb of steps to add
    '''
    with open(trainer_config_path) as f:
         list_doc = yaml.load(f)

    list_doc['AnimalAI']['max_steps'] = float(list_doc['AnimalAI']['max_steps']) + steps_add

    with open(trainer_config_path, "w") as f:
        yaml.dump(list_doc, f)
        
#logs_dir = "summaries/"
#os.makedirs(logs_dir, exist_ok=True)

def train_protagonist(arena_config, n_steps_trainer = 1.0e4, base_port_protagonist = 4000, run_id_protagonist = "protagonist", load_model = False, trainer_config_path = "configurations/train_ml_agents_config_ppo_10fs.yaml"):
    
    '''
    Function to train a protagonist AnimalAI training.
    Args:
        arena_config: ArenaConfig object. Environment where the protagonist will be trained
        n_steps_trainer: n_steps used for training
        base_port_protagonist: port where the animalai train will run
        run_id_protagonist: id used to save the model checkpoints and training performance
        load_model: bool. If True, it loads an existing model with id run_id_protagonist. Otherwise, it starts a new training
    '''
    
    environment_path = "../env/AnimalAI"
    logs_dir = "summaries/"
    trainer_config_path = trainer_config_path
    
    if load_model:
        update_trainer(trainer_config_path, steps_add=n_steps_trainer)
    else:
        reset_trainer(trainer_config_path, steps=n_steps_trainer)
    
    args = RunOptionsAAI(
        trainer_config=load_config(trainer_config_path),
        env_path=environment_path,
        run_id=run_id_protagonist,
        base_port=base_port_protagonist,
        seed = 0,
        load_model=load_model,
        train_model=True,
        arena_config=arena_config 
    )
    run_training_aai(0, args)
    
    data_path = 'summaries/' + run_id_protagonist + '_AnimalAI.csv'
    df = pd.read_csv(data_path)
    print('PROTAGONIST: ')
    print('Steps: ', df.loc[0, 'Steps'], ' Cumulative reward: ', df.loc[0, 'Environment/Cumulative Reward'], \
     ' Episode Length: ', df.loc[0, 'Environment/Episode Length'])
    print('Steps: ', df.loc[int(len(df)/2), 'Steps'], ' Cumulative reward: ', df.loc[int(len(df)/2), 'Environment/Cumulative Reward'], \
         ' Episode Length: ', df.loc[int(len(df)/2), 'Environment/Episode Length'])
    print('Steps: ', df.loc[len(df)-1, 'Steps'], ' Cumulative reward: ', df.loc[len(df)-1, 'Environment/Cumulative Reward'], \
         ' Episode Length: ', df.loc[len(df)-1, 'Environment/Episode Length'])
    
    return df.loc[len(df)-1, 'Environment/Cumulative Reward']

def train_antagonist(arena_config, n_steps_trainer = 1.0e4, base_port_antagonist = 5000, run_id_antagonist = "antagonist", load_model = False, trainer_config_path = "configurations/train_ml_agents_config_ppo_10fs_2.yaml"):
    
    '''
    Function to train a antagonist AnimalAI training.
    Args:
        arena_config: ArenaConfig object. Environment where the antagonist will be trained
        n_steps_trainer: n_steps used for training
        base_port_antagonist: port where the animalai train will run
        run_id_antagonist: id used to save the model checkpoints and training performance
        load_model: bool. If True, it loads an existing model with id run_id_antagonist. Otherwise, it starts a new training
    '''
    
    environment_path = "../env/AnimalAI"
    logs_dir = "summaries/"
    trainer_config_path = trainer_config_path
    
    if load_model:
        update_trainer(trainer_config_path, steps_add=n_steps_trainer)
    else:
        reset_trainer(trainer_config_path, steps=n_steps_trainer)
    
    args = RunOptionsAAI(
        trainer_config=load_config(trainer_config_path),
        env_path=environment_path,
        run_id=run_id_antagonist,
        seed = 1,
        base_port=base_port_antagonist,
        load_model=load_model,
        train_model=True,
        arena_config=arena_config 
    )
    run_training_aai(1, args)
    
    data_path = 'summaries/' + run_id_antagonist + '_AnimalAI.csv'
    df = pd.read_csv(data_path)
    print('ANTAGONIST: ')
    print('Steps: ', df.loc[0, 'Steps'], ' Cumulative reward: ', df.loc[0, 'Environment/Cumulative Reward'], \
     ' Episode Length: ', df.loc[0, 'Environment/Episode Length'])
    print('Steps: ', df.loc[int(len(df)/2), 'Steps'], ' Cumulative reward: ', df.loc[int(len(df)/2), 'Environment/Cumulative Reward'], \
         ' Episode Length: ', df.loc[int(len(df)/2), 'Environment/Episode Length'])
    print('Steps: ', df.loc[len(df)-1, 'Steps'], ' Cumulative reward: ', df.loc[len(df)-1, 'Environment/Cumulative Reward'], \
         ' Episode Length: ', df.loc[len(df)-1, 'Environment/Episode Length'])
    
    return df.loc[len(df)-1, 'Environment/Cumulative Reward']