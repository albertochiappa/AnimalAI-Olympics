import gym
import gym_example
from utils import *
import sys

import os
import shutil
import ray

from ray.tune.registry import register_env
from gym_example.envs.adversarial_v62 import Adversarial_v62
import ray.rllib.agents.ppo as ppo

if __name__ == "__main__":
    
    sys.setrecursionlimit(10000)

    chkpt_root = "tmp/test"
    chkpt_file = chkpt_root + '/checkpoint_105/checkpoint-105'

    #shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    ray.init(ignore_reinit_error=True, local_mode=True)
    
    select_env = "adversarial-v62"
    register_env(select_env, lambda config: Adversarial_v62())

    config = ppo.DEFAULT_CONFIG.copy()

    config['num_workers'] = 0
    config["log_level"] = "WARN"
    config["train_batch_size"] = 1
    config["sgd_minibatch_size"] = 1
    config["num_sgd_iter"] = 1
    config["timesteps_per_iteration"] = 4
    config["rollout_fragment_length"] = 1
    
    agent = ppo.PPOTrainer(config, env=select_env)
    agent.restore(chkpt_file)
    
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 1000
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))

