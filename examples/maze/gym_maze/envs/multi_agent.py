from typing import Tuple, Dict, List
import gym

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pickle


class BasicMultiAgent(MultiAgentEnv):
    def __init__(self, env_config, maze_generator):
    #def __init__(self):
        
        self.agents = [gym.make(env_config["env_id"], maze_generator=maze_generator)\
               for _ in range(env_config["num_agents"])]
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[1].action_space
        self.reset()
        
    def reset(self):
        self.resetted = True
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}
    
    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info
    
    
class AlternateMultiAgent(MultiAgentEnv):
    "Alternate training of protagonist and antagonist"

    def __init__(self, env_config, maze_generator):
        self.agents = [gym.make(env_config["env_id"], maze_generator=maze_generator)\
               for _ in range(env_config["num_agents"])]
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.i = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}

    def reset(self):
        self.i = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        # collect observations and rewards for all
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action) 
            
        # only retyrn for first agent
        obs = {self.i: self.last_obs[self.i]} 
        rew = {self.i: self.last_rew[self.i]}
        done = {self.i: self.last_done[self.i]}
        info = {self.i: self.last_info[self.i]}
        
        done["__all__"] = False
        
        if done[self.i]:
            if self.i == 0: 
                self.i = 1
            else:
                done["__all__"] = True

        return obs, rew, done, info
    
class AdvPro(MultiAgentEnv):

    def __init__(self, adversary, protagonist, save = False, file_name = []):
        
        self.agents = [adversary, protagonist]
        
        self.observation_space = {0: self.agents[0].observation_space, 1: self.agents[1].observation_space}
        self.action_space = {0: self.agents[0].action_space, 1: self.agents[1].action_space}
        self.i = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.MAX_STEPS = 250
        
        self.n_episodes = 0
        self.file_name = file_name
        self.save = save

    def reset(self):
        self.i = 0
        self.count = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        # collect observations and rewards for all
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action) 
            
        # only retyrn for first agent
        obs = {self.i: self.last_obs[self.i]} 
        rew = {self.i: self.last_rew[self.i]}
        done = {self.i: self.last_done[self.i]}
        info = {self.i: self.last_info[self.i]}
        
        done["__all__"] = False
        
        if self.i == 1:
            self.count += 1
            
        if self.count == self.MAX_STEPS:
            done["__all__"] = True
        
        if done[self.i]:
            if self.i == 0: 
                self.i = 1
                maze = self.agents[0].state['image']
                self.agents[1].reset(maze)
                
                if self.save:
                    with open(self.file_name+str(self.n_episodes)+'.pkl','wb') as f:
                        pickle.dump(maze, f)
                
                self.n_episodes +=1 
            else:
                done["__all__"] = True
                
        

        return obs, rew, done, info
    
class AdvPro2(MultiAgentEnv):

    def __init__(self, adversary, protagonist, save = False, file_name = [], MAX_STEPS=250):
        
        self.agents = [adversary, protagonist]
        
        self.observation_space = {0: self.agents[0].observation_space, 1: self.agents[1].observation_space}
        self.action_space = {0: self.agents[0].action_space, 1: self.agents[1].action_space}
        self.i = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.MAX_STEPS = MAX_STEPS
        self.reward_adv = 0
        self.pro_finished = False
        
        self.n_episodes = 0
        self.file_name = file_name
        self.save = save

    def reset(self):
        self.i = 0
        self.count = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.pro_finished = False
        self.reward_adv = 0
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        # collect observations and rewards for all
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action) 
            
        if not self.pro_finished: # if there is no reward for the adversary
            obs = {self.i: self.last_obs[self.i]} 
            rew = {self.i: self.last_rew[self.i]}
            done = {self.i: self.last_done[self.i]}
            info = {self.i: self.last_info[self.i]}

            done["__all__"] = False

            if self.i == 1:
                self.reward_adv += self.last_rew[self.i]
                self.count += 1
                if done[self.i] or self.count == self.MAX_STEPS: # finish training protagonist
                    done[self.i] = True
                    self.pro_finished = True
                    self.i = 0 # go back to the adversary agent
                    
            elif self.i == 0:
                if info[self.i]["ready"] == "yes": # the adversary is ready, has created the env
                    maze = self.agents[0].state['image']
                    self.agents[1].reset(maze)

                    if self.save:
                        with open(self.file_name+str(self.n_episodes)+'.pkl','wb') as f:
                            pickle.dump(maze, f)
                    self.n_episodes += 1
                    
                    self.i = 1
            
                
        elif self.pro_finished: # if there is reward , we set it to the reward of the adversary
            obs = {self.i: self.last_obs[self.i]} 
            self.reward_adv = 0
            rew = {self.i: self.reward_adv} # the reward it gets is the reward the adversary gets
            done = {self.i: self.last_done[self.i]}
            info = {self.i: self.last_info[self.i]}
            done["__all__"] = True
                

        return obs, rew, done, info

class PAIRED(MultiAgentEnv):

    def __init__(self, adversary, protagonist, antagonist, save = False, file_name = [], MAX_STEPS = 250, sum_rewards = False):
        
        self.agents = [adversary, protagonist, antagonist]
        
        self.observation_space = {0: self.agents[0].observation_space,\
                                  1: self.agents[1].observation_space, 2: self.agents[2].observation_space}
        self.action_space = {0: self.agents[0].action_space, 1: self.agents[1].action_space, \
                             2: self.agents[2].action_space}

        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.MAX_STEPS = MAX_STEPS
        self.sum_rewards = sum_rewards
        
        self.env_ready = False
        self.pro_finished = False
        self.ant_finished = False
        self.pro_reward = 0
        self.ant_reward = 0
        
        self.save = save
        self.file_name = file_name
        self.n_episodes = 0
        self.info_episode = []

    def reset(self):
        self.count = 0
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.info_episode = []
        
        self.env_ready = False
        self.pro_finished = False
        self.ant_finished = False
        self.pro_reward = 0
        self.ant_reward = 0
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        
        obs, rew, done, info = {}, {}, {}, {}
        
        # collect observations and rewards 
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action) 
            
        # if the maze is not ready --> adversary generates maze
        if not self.env_ready:
            obs[0] = self.last_obs[0]
            rew[0] = self.last_rew[0]
            done[0] = self.last_done[0]
            info[0] = self.last_info[0]

            done["__all__"] = False
                
            if info[0]["ready"] == "yes": 
                # if the maze is ready set maze to true
                self.env_ready = True
                #print("environment ready")
                # update both agents
                
                maze = self.agents[0].state['image']

                if self.save:
                    self.info_episode.append(maze)
                    #with open(self.file_name+str(self.n_episodes)+'.pkl','wb') as f:
                     #   pickle.dump(maze, f)
                self.n_episodes += 1
                    
                self.agents[1].reset(maze) 
                self.agents[2].reset(maze) 
                
        elif self.env_ready and self.pro_finished and self.ant_finished:
            obs[0] = self.last_obs[0]
            if self.sum_rewards:
                rew[0] = self.ant_reward + self.pro_reward
            else:
                rew[0] = self.ant_reward - self.pro_reward
            #rew[0] =  - (self.agents[0].shortest_path_length - 8)
            done[0] = self.last_done[0]
            info[0] = self.last_info[0]
            
            if self.save:
                self.info_episode.append(self.ant_reward)
                self.info_episode.append(self.pro_reward)
                self.info_episode.append(rew[0])

                with open(self.file_name+str(self.n_episodes)+'.pkl','wb') as f:
                    pickle.dump(self.info_episode, f)

            done["__all__"] = True
            
        else: # environment ready and one of the agents has not finished
            done["__all__"] = False
            
            self.count += 1
            
            if not self.pro_finished:
                obs[1] = self.last_obs[1]
                rew[1] = self.last_rew[1]
                done[1] = self.last_done[1]
                info[1] = self.last_info[1]
                
                self.pro_reward += rew[1]
                
                if self.count == self.MAX_STEPS:
                    done[1] = True
                
                if done[1]:
                    self.pro_finished = True
                    #print("protagonist ready")
                
                
            if not self.ant_finished:
                obs[2] = self.last_obs[2]
                rew[2] = self.last_rew[2]
                done[2] = self.last_done[2]
                info[2] = self.last_info[2]
                
                self.ant_reward += rew[2]
                
                if self.count == self.MAX_STEPS:
                    done[2] = True
                
                if done[2]:
                    self.ant_finished = True
                    #print("antagonist ready")           

        return obs, rew, done, info
