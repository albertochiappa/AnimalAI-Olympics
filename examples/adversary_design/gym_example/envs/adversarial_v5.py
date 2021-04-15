import gym
from gym.utils import seeding
import numpy as np
from random import randint

import networkx as nx
from networkx import grid_graph

from utils import *

class Adversarial_v5(gym.Env):
    ''' Grid world where an adversary build the environment the agent plays.
    The adversary places the goal, agent, and up to n_clutter blocks in sequence.
    The action dimension is the number of squares in the grid, and each action
    chooses where the next item should be placed. '''

    def __init__(self, n_clutter=1, size=38):
        '''Initializes environment in which adversary places goal, agent, obstacles.
        Args:
          n_clutter: The maximum number of obstacles the adversary can place.
          size: The number of tiles across one side of the grid; i.e. make a size x size grid.
          max_steps: The maximum number of steps that can be taken before the episode terminates.
        '''
        # define params
        self.n_clutter = n_clutter
        # Add two actions for placing the agent and goal.
        self.MAX_STEPS = self.n_clutter + 2
        self.size=size
        
        # to train protagonist and antagonist
        self.n_episodes = 0
        self.load_model = True
        self.base_port_protagonist = 4000
        self.base_port_antagonist = 5000
        self.run_id_protagonist = "self_control_curriculum_protagonist"
        self.run_id_antagonist = "self_control_curriculum_antagonist"
        self.trainer_config_path_pro = "configurations/train_ml_agents_config_ppo_pro.yaml"
        self.trainer_config_path_ant = "configurations/train_ml_agents_config_ppo_ant.yaml"
        self.n_steps_trainer = 10000

        # Create spaces for adversary agent's specs: all spaces where it can place an object
        self.action_space = gym.spaces.Discrete(size**2)
        
        # observation step: image + time step
        self.image_space = gym.spaces.Box(low=0,high=3,shape=(self.size, self.size),dtype=np.float16)
        self.ts_space = gym.spaces.Box(low=0, high=self.MAX_STEPS, shape=(1,), dtype='uint8')
        
        self.observation_space = gym.spaces.Dict(
            {'image': self.image_space,
             'time_step': self.ts_space})
        
        self.seed()
        
        self.reset()
        
    def reset(self):
        self.agent_start_pos = -1
        self.goal_pos = -1
        self.count = 0    
        self.image_space = np.zeros((self.size, self.size))
        self.state = {
            'image': self.image_space,
            'time_step': [self.count]
        }
        self.reward = 0
        self.done = False
        self.info = {}
        self.wall_locs = []
        self.graph = grid_graph(dim=[self.size, self.size])
        self.passable = -1
        self.distance_to_goal = None
        self.shortest_path_length = None
        
        return self.state
        
    def compute_shortest_path(self):
        if self.agent_start_pos is None or self.goal_pos is None:
            return

        self.distance_to_goal = abs(
            self.goal_pos[0] - self.agent_start_pos[0]) + abs(
                self.goal_pos[1] - self.agent_start_pos[1])

        # Check if there is a path between agent start position and goal
        self.passable = nx.has_path(
            self.graph,
            source=(self.agent_start_pos[0], self.agent_start_pos[1]),
            target=(self.goal_pos[0], self.goal_pos[1]))
        if self.passable:
          # Compute shortest path
            self.shortest_path_length = nx.shortest_path_length(
              self.graph,
              source=(self.agent_start_pos[0], self.agent_start_pos[1]),
              target=(self.goal_pos[0], self.goal_pos[1]))
        else:
          # Impassable environments have a shortest path length 1 longer than the longest possible path
            self.shortest_path_length = (self.size-1) * (self.size-1) + 1
        
    def step (self, action):
        if self.done:
          # should never reach this point
            print("EPISODE DONE!!!")
        elif self.count == self.MAX_STEPS:
            
            for w in self.wall_locs:
                self.graph.remove_node(w)
            self.compute_shortest_path()
            
            print('Shortest path: ', self.shortest_path_length )
            
            # padd with zeros observation space (so before we did not consider walls)
            matrix = np.zeros((40, 40))
            matrix[1:39,1:39] = self.image_space 
            
            # create AnimalAI environment configuration from matrix
            arena_config = matrix2arena(matrix)
            
            # visualize environment
            visualize(matrix) 
            
            # run protagonist training
            reward_protagonist = train_protagonist(arena_config=arena_config, n_steps_trainer=self.n_steps_trainer, \
                                       base_port_protagonist=self.base_port_protagonist, run_id_protagonist = self.run_id_protagonist,  \
                                       load_model=self.load_model, trainer_config_path = self.trainer_config_path_pro)
            # run antagonist training
            reward_antagonist = train_antagonist(arena_config=arena_config, n_steps_trainer=self.n_steps_trainer, \
                                       base_port_antagonist=self.base_port_antagonist, run_id_antagonist = self.run_id_antagonist,\
                                       load_model=self.load_model, trainer_config_path = self.trainer_config_path_ant)
            
            # update trainer (add steps)
            self.n_episodes += 1
            
            self.reward = reward_antagonist - reward_protagonist
            
            self.done = True
              
        else:
            try:
                assert self.action_space.contains(action)
                
                x = int(action % self.size) 
                y = int(action / self.size) 

                should_choose_goal = self.count == 0
                should_choose_agent = self.count == 1
                
                if should_choose_goal:
                    self.image_space[x, y] = 1
                    self.goal_pos = [x, y]
                elif should_choose_agent:
                    if self.image_space[x, y] != 1:
                        self.image_space[x, y] = 2
                    else:
                        while self.image_space[x, y] == 1:
                            l = randint(0, self.size**2)
                            x = int(l % self.size) 
                            y = int(l / self.size)
                        self.image_space[x, y] = 2
                    self.agent_start_pos = [x, y]    
                else:
                    if self.image_space[x, y] != 1 and self.image_space[x, y] != 2:
                        self.image_space[x, y] = 3
                        if (x, y) not in self.wall_locs:
                            self.wall_locs.append((x, y))
                self.count += 1
                
            except AssertionError:
                print("INVALID ACTION", action)  
                
         
        self.state = {
            'image': self.image_space,
            'time_step': [self.count]
        }
        
        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)  
        
        return self.state, self.reward, self.done, self.info
    
    def render(self, mode="human"):
        s = "reward: {:2d}  info: {}"
        print(s.format(self.reward, self.info))
        
    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s)."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]