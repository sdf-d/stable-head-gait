import numpy as np
import gym
import roboschool
#import baselines

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





class PPO_Class:


    #NUM_PROCESSES = 8
    ENV_NAME = "RoboschoolHalfCheetah6-v1"#"RoboschoolHumanoid-v1"
    SAVE_PATH = "./modelpath/cheetah6/model_d3_1_40_40__rew03_000headdist2_newh2_0301_asdasd"
    RETURN_SAVE_PATH = "./modelpath/cheetah6/ret_d3_1_40_40__rew03_000headdist2_newh2_0301_asdasd"

    STEPSIZE = 256
    NUM_STEPS = 256
    NUM_EPOCHS = 10
    HIDDEN_SIZE = 256



    GAMMA = 0.99
    LAMBDA = 0.97

    PI_LR = 1e-4
    V_LR = 1e-4


    PPO_UPDATES_PER_EPOCH = 80
    #UPE: Updates Per Epoch
    PI_UPE = 50
    V_UPE = 50
    MAX_KL = 0.015 #cut policy learning after reaching this
    TIMES_TRAIN = 6000
    STEPS_PER_EPOCH = 4000
    MAX_EP_LEN = 1000
    CLIP_PAR = 0.2

    SEED = 344121





    class IteratePPO:

        def __iter__(self):
            self.ppo_struct = PPO_Struct()


    def __init__(self):
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        self.env = gym.make(self.ENV_NAME)
        self.ppo_struct = PPO_Struct(self.STEPS_PER_EPOCH, self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.moddel = MLP_actor_critic_2(self.env.observation_space.shape[0], self.env.action_space.shape[0])

        actorparams = list(self.moddel.actor.parameters()) + [self.moddel.log_std]

        self.opt_pi = torch.optim.Adam(actorparams, lr=self.PI_LR)
        self.opt_v = torch.optim.Adam(self.moddel.critic.parameters(), lr=self.V_LR)



        self.epochrewards = np.zeros(self.TIMES_TRAIN, dtype=np.float32)


            #break
    def policy_loss(self, obs_t,act_t,adv_t,logprob_old_t):
        #obs, act, adv, logp_old from ppodata

        logprob = self.moddel.logprob(obs_t, act_t)
        logprobdiff = logprob - logprob_old_t
        probratio = torch.exp(logprobdiff)
        tomin1 = probratio * adv_t
        clipped = torch.clamp(probratio, 1-self.CLIP_PAR, 1+self.CLIP_PAR)
        tomin2 =  clipped * adv_t
        retloss = -torch.min(tomin1, tomin2).mean()

        #use kl early stopping for better performance
        #from spinup
        approx_kl_t = (logprob_old_t-logprob).mean()
        approx_kl = approx_kl_t.item()


        return retloss, approx_kl

    def value_loss(self, obs_t, retval_t):
        #obs_t, ret_v from ppodata

        #quadratic loss
        retloss = ((self.moddel.critic(obs_t) - retval_t)**2).mean()

        return retloss


    def ppo_update(self):
        #get data as tensors

        obs_t = torch.as_tensor(self.ppo_struct.states, dtype=torch.float32)
        act_t = torch.as_tensor(self.ppo_struct.actions, dtype=torch.float32)
        ret_t = torch.as_tensor(self.ppo_struct.returns, dtype=torch.float32)
        log_p_t = torch.as_tensor(self.ppo_struct.log_probs, dtype=torch.float32)
        adv_t = torch.as_tensor(self.ppo_struct.advantages, dtype=torch.float32)

        p_loss_d = self.policy_loss(obs_t, act_t, adv_t, log_p_t)
        v_loss_d = self.value_loss(obs_t, ret_t).item()

        print("policy loss: ", p_loss_d)
        print("value loss: ", v_loss_d)


        for i in range(self.PI_UPE):
            self.opt_pi.zero_grad()
            pi_loss, approx_kl = self.policy_loss(obs_t, act_t, adv_t, log_p_t)
            if approx_kl > self.MAX_KL:
                print("max kl reached at step: ", i)
                break
            pi_loss.backward()
            self.opt_pi.step()

        for i in range(self.V_UPE):

            self.opt_v.zero_grad()
            v_loss = self.value_loss(obs_t, ret_t)

            v_loss.backward()
            self.opt_v.step()


        #plotdata.logepoch(P_loss=p_loss_d, V_loss=v_loss_d)

    def train(self):
        obs = self.env.reset()
        ep_start = 0
        ep_end = -1
        cur_ret = 0
        cur_len = 0



        for e1 in range(self.TIMES_TRAIN):
            print("Epoch: ", e1)
            avg_reward = 0
            num_eps = 0




            #PART 1: gather experience
            for e2 in range(self.STEPS_PER_EPOCH):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                act, val, logp = self.moddel.act(obs_t)

                new_obs, rew, done, _ = self.env.step(act)
                cur_ret += rew
                cur_len += 1

                #store in ppo_struct--
                self.ppo_struct.states[e2] = obs
                self.ppo_struct.actions[e2] = act
                self.ppo_struct.rewards[e2] = rew
                self.ppo_struct.values[e2] = val
                self.ppo_struct.log_probs[e2] = logp
                self.ppo_struct.dones[e2] = done

                obs = new_obs

                #if episode terminated
                if done or cur_len == self.MAX_EP_LEN or e2 == self.STEPS_PER_EPOCH-1:

                    #if terminated normally
                    if done:
                        val = 0
                        avg_reward += cur_ret
                        num_eps += 1

                    #if terminated before episode ends, use estimated value from network
                    if cur_len == self.MAX_EP_LEN or e2 == self.STEPS_PER_EPOCH-1:
                        obs_t = torch.as_tensor(obs, dtype=torch.float32)
                        aa , val , aa = self.moddel.act(obs_t)

                    #self.ppo_struct.last_val(val)

                    #print("return: ", cur_ret)

                    obs = self.env.reset()


                    cur_ret = 0
                    cur_len = 0



                    discount_rew1 = np.append(self.ppo_struct.rewards[ep_start:e2], val)
                    discount_val1 = np.append(self.ppo_struct.values[ep_start:e2], val)

                    delta_vals = discount_rew1[:-1] + self.GAMMA * discount_val1[1:] - discount_val1[:-1]
                    self.ppo_struct.advantages[ep_start:e2] = times_cumulate(delta_vals, self.GAMMA * self.LAMBDA)

                    self.ppo_struct.returns[ep_start:e2] =  times_cumulate(discount_rew1, self.LAMBDA)[:-1]





                    ep_start = e2
                    ep_end = e2




            avg_reward = 0 if num_eps == 0 else avg_reward/num_eps
            self.epochrewards[e1] = avg_reward
            print("Avg Rew: ", avg_reward)
            avg_reward = 0
            num_eps = 0
            #PART 2: train networks
            self.ppo_struct.advantages = utils.normalize(self.ppo_struct.advantages)
            ep_start = 0
            self.ppo_update()



            if e1 % 20 == 0:
                torch.save(self.moddel.state_dict(), self.SAVE_PATH)
                np.save(self.RETURN_SAVE_PATH, self.epochrewards)


            #rinse and repeat

            if e1%2000 == 0:
                pdb.set_trace()

if __name__ == "__main__":

    ppo_class = PPO_Class()
    ppo_class.train()
