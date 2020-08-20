from OpenGL import GLU
import numpy as np
import gym
import roboschool

import torch
import torch.nn as nn
import torch.distributions

import env_maker

from ppo_struct import PPO_Struct
from ppo_struct import Step_Struct
from nn_models import MLP_actor_critic
from nn_models import MLP_actor_critic_2

from utils import times_cumulate
import utils

import pdb
import ipdb


from timeit import default_timer as timer
import time


LOAD_DIR = "./modelpath/cheetah6/model_d3_1_40_40__rew03_headdist2_newh2_0301"
ENV_NAME = "RoboschoolHalfCheetah6-v1"


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    #env = gym.wrappers.Monitor(env, "./vid", video_callable=False,force=True)

    moddel = MLP_actor_critic_2(env.observation_space.shape[0], env.action_space.shape[0])
    moddel.load_state_dict(torch.load(LOAD_DIR))


    obs = env.reset()
    ep_start = 0
    ep_end = -1
    cur_ret = 0
    cur_len = 0

    done = False
    env.render()
    #ipdb.set_trace()



    while not done:
        #print(obs)

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        act, val, logp = moddel.act(obs_t)

        new_obs, rew, done, _ = env.step(act)
        cur_ret += rew
        cur_len += 1


        obs = new_obs


        env.render()
        time.sleep(0.03)
