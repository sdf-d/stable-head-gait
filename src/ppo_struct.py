import numpy as np
import torch
import torch.nn as nn

import utils

import pdb


class PPO_Struct:

	def __init__(self, bufsize,statedim,actiondim):


		self.states = np.zeros((bufsize,statedim), dtype=np.float32)
		self.actions = np.zeros((bufsize,actiondim), dtype=np.float32)
		self.values = np.zeros(bufsize, dtype=np.float32)
		self.dones = np.zeros(bufsize, dtype=np.float32)
		self.log_probs = np.zeros(bufsize, dtype=np.float32)
		self.rewards = np.zeros(bufsize, dtype=np.float32)
		self.returns = np.zeros(bufsize, dtype=np.float32)
		self.advantages = np.zeros(bufsize, dtype=np.float32)
		self.gaes = np.zeros(bufsize, dtype=np.float32)

	def steps_combine(self):
		self.returns = torch.cat(self.returns).detach()
		self.log_probs = torch.cat(self.log_probs).detach()
		self.values = torch.cat(self.values).detach()
		self.states = torch.cat(self.states).detach()
		self.actions = torch.cat(self.actions).detach()
