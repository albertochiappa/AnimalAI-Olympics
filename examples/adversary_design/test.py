import gym
import gym_example
from utils import *

import os
import shutil
import ray

from ray.tune.registry import register_env
from gym_example.envs.adversarial_v4 import Adversarial_v4
import ray.rllib.agents.ppo as ppo

if __name__ == "__main__":

    chkpt_root = "tmp/test"

    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    ray.init(ignore_reinit_error=True, local_mode=True)
    
    select_env = "adversarial-v4"
    register_env(select_env, lambda config: Adversarial_v4())

    config = ppo.DEFAULT_CONFIG.copy()

    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=select_env)
    
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 10
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

