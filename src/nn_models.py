import torch
import torch.nn as nn
from torch.distributions import Normal


import pdb
import ipdb

class MLP_actor_critic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(MLP_actor_critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


class Critic1(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=100):
        super(Critic1, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, (int)(hidden_size/2)),
            nn.Tanh(),
            nn.Linear((int)(hidden_size/2), (int)(hidden_size/4)),
            nn.Tanh(),
            nn.Linear((int)(hidden_size/4), 1),
        )

    def forward(self, obs):
        return torch.squeeze(self.network(obs), -1)


class MLP_actor_critic_2(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=100, std=-0.5):
        super(MLP_actor_critic_2, self).__init__()

        self.critic = Critic1(num_inputs, num_outputs, hidden_size)
        """nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, (int)(hidden_size/2)),
            nn.Tanh(),
            nn.Linear((int)(hidden_size/2), (int)(hidden_size/4)),
            nn.Tanh(),
            nn.Linear((int)(hidden_size/4), 1),
        )"""

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, (int)(hidden_size/2)),
            nn.Tanh(),
            nn.Linear((int)(hidden_size/2), (int)(hidden_size/4)),
            nn.Tanh(),
            nn.Linear((int)(hidden_size/4), num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)

        pdb.set_trace()
        print("Why am I here")

        return dist, value

    def get_dist(self,obs):
        return Normal(self.actor(obs), torch.exp(self.log_std))

    def get_log_prob(self,dist,action):
        return dist.log_prob(action).sum(axis=-1)

    def logprob(self,obs,act):
        dist = self.get_dist(obs)
        ret_logp = self.get_log_prob(dist,act)
        return ret_logp


    def act(self, obs):
        with torch.no_grad():
            policydist = self.get_dist(obs)

            sample_a = policydist.sample()
            logprob = self.get_log_prob(policydist, sample_a)
            val = self.critic(obs)
        return sample_a.numpy(), val.numpy(), logprob.numpy()
