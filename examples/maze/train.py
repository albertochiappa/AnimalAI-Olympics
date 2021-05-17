from argparse import ArgumentParser

from gym_maze.envs import MazeEnv2, Adversary, PAIRED
import matplotlib.pyplot as plt
import gym

import os
import pandas as pd
import shutil
import ray

from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

def parse_args():
    parser = ArgumentParser(description='PAIRED training')
    
    parser.add_argument(
        '-n_clutter', '--n_clutter',
        type=int, default=10,
        help='Number of walls.'
    )
    
    parser.add_argument(
        '-size', '--size',
        type=int, default=6,
        help='Size of the maze.'
    )
    
    parser.add_argument(
        '-save', '--save',
        type=bool, default=False,
        help='Save the maze and rewards.'
    )
    
    parser.add_argument(
        '-file_name', '--file_name',
        type=str, default=None,
        help='Directory where to save the files.'
    )
    
    parser.add_argument(
        '-sum_rewards', '--sum_rewards',
        type=bool, default=False,
        help='Whether to sum or not the rewards obtained by protagonist and antagonist.'
    )
    
    parser.add_argument(
        '-max_steps', '--max_steps',
        type=int, default=300,
        help='Max number of steps of the protagonist and antagonist.'
    )
    
    parser.add_argument(
        '-chkpt_root', '--chkpt_root',
        type=str, default='tmp/trial',
        help='Directory where ray saves the training checkpoints.'
    )
    
    parser.add_argument(
        '-n_iter', '--n_iter',
        type=int, default=100,
        help='Number of training iterations.'
    )


    return parser.parse_args()

def run_one_episode_adv(env):
    '''Given a gym environment (env) it computer one episode and return cumulative reward.
    Used to check if our gym environments work'''
    
    env.reset()
    sum_reward = 0
    
    steps = []
    frames = []
    
    steps.append(env.state.copy())


    while not env.done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        steps.append(env.state.copy())
        sum_reward += reward
        
    return env

if __name__ == '__main__':
    
    args = parse_args()
    
    n_clutter = args.n_clutter
    size = args.size
    save = args.save
    file_name = args.file_name
    sum_rewards = args.sum_rewards
    MAX_STEPS = args.max_steps
    
    chkpt_root = args.chkpt_root
    n_iter = args.n_iter

    # init ray
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    ray.init(ignore_reinit_error=True, local_mode=True)
    
    # initialize adversary, antagonist and protagonist
    adversary = Adversary(n_clutter=n_clutter, size=size)
    adversary = run_one_episode_adv(adversary)
    protagonist = MazeEnv2(adversary.state['image'])
    antagonist = MazeEnv2(adversary.state['image'])
    adversary = Adversary(n_clutter=n_clutter, size=size)

    # define policies and policy mapping function
    policies = {
            "ppo_policy_0": (ppo.PPOTFPolicy, adversary.observation_space, adversary.action_space, {}),
            "ppo_policy_1": (ppo.PPOTFPolicy, protagonist.observation_space, protagonist.action_space, {}),
            "ppo_policy_2": (ppo.PPOTFPolicy, antagonist.observation_space, antagonist.action_space, {})
        }

    def policy_mapping_fn(agent_id):
        if agent_id == 0:
            return "ppo_policy_0"
        elif agent_id == 1:
            return "ppo_policy_1"
        elif agent_id == 2:
            return "ppo_policy_2"
    
    
    # register the environment
    select_env = 'Paired-v0'
    env_config = PAIRED(adversary, protagonist, antagonist, save=save, \
                        file_name=file_name, MAX_STEPS=MAX_STEPS, sum_rewards=sum_rewards)
    register_env(select_env, lambda config: env_config)
    
    # create agent
    agent = ppo.PPOTrainer(
        env=select_env,
        config={
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["ppo_policy_0", "ppo_policy_1", "ppo_policy_2"],
            },
            "vf_loss_coeff": 0.01,
            "framework": "tf",
        })
    
    # train and print training results
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    os.mkdir(file_name)
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        min_reward = result['policy_reward_min']
        mean_reward  = result['policy_reward_mean']
        max_reward = result['policy_reward_max']
        id0 = "ppo_policy_0"
        id1 = "ppo_policy_1"
        id2 = "ppo_policy_2"
        print(n + 1, 'Agent -  min - mean - max ')
        print('-----------------------------------------------------')
        print('Adv ', round(min_reward[id0], 2), '  ', round(mean_reward[id0], 2), '  ', round(max_reward[id0], 2))
        print('Pro ', round(min_reward[id1], 2), '  ', round(mean_reward[id1], 2), '  ', round(max_reward[id1], 2))
        print('Ant ', round(min_reward[id2], 2), '  ', round(mean_reward[id2], 2), '  ', round(max_reward[id2], 2))
        print(' ')