from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import copy, time
from pathlib import Path


from torch import nn
from collections import namedtuple
# from torch.distributions import Categorical
from torch.distributions import Normal, Independent

import pickle, os, random, torch

from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt

import pdb

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

#class DDPGExtension(DDPGAgent):
    #pass
    
class Policy_new(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


# Critic class. The critic is represented by a neural network.
class Critic_new(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]
    
    
    
class DDPGExtension(DDPGAgent):
    def __init__(self, config=None, mode = None):
        super(DDPGExtension, self).__init__(config)
        self.update_counter = 0
        # Additional initialization or modifications can be done here if needed.
        state_dim = self.observation_space_dim
        self.action_dim = self.action_space_dim
        
        if mode == 'difficult':
            self.q2 = Critic_new(state_dim, self.action_dim).to(self.device)
        else:
            self.q2 = Critic(state_dim, self.action_dim).to(self.device)
            
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=float(self.lr))
        
        # Other TD3-specific attributes, if any
        self.policy_delay = 8
        self.policy_noise = 0.3
        #self.noise_clip = self.cfg.noise_clip        
        
        # Reloading the new configuration of policy and critic network ###########
        if mode == 'difficult':
            self.pi = Policy_new(state_dim, self.action_dim, self.max_action).to(self.device)
        else: 
            self.pi = Policy(state_dim, self.action_dim, self.max_action).to(self.device)
            
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=float(self.lr))
        
        if mode == 'difficult':
            self.q = Critic_new(state_dim, self.action_dim).to(self.device)
        else:
            self.q = Critic(state_dim, self.action_dim).to(self.device)
            
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=float(self.lr))    

    def _update(self):
        self.update_counter += 1  # Increment the update counter
        #print('update_counter', self.update_counter)
        # get batch data
        batch = self.buffer.sample(self.batch_size, device=self.device)
        state, action, next_state, reward, not_done = batch.state, batch.action, batch.next_state, batch.reward, batch.not_done

        # compute current q values for both critics
        #pdb.set_trace()
        current_q1 = self.q(state, action)
        current_q2 = self.q2(state, action)

        # compute target q values using the target actor and both target critics
        with torch.no_grad():
            next_action = self.pi_target(next_state)
            #noise = torch.clamp(torch.normal(0, self.policy_noise, size=next_action.shape).to(self.device), -self.noise_clip, self.noise_clip)
            #next_action = torch.clamp(next_action + noise, -self.max_action, self.max_action)
            
            next_action = next_action + torch.normal(0, self.policy_noise, size=next_action.shape).to(self.device)
            next_action = torch.clamp(next_action, -self.max_action, self.max_action)
            
            target_q1 = self.q_target(next_state, next_action) * not_done
            target_q2 = self.q2_target(next_state, next_action) * not_done
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * target_q

        # compute critic losses for both critics
        critic_loss1 = F.mse_loss(current_q1, target_q)
        critic_loss2 = F.mse_loss(current_q2, target_q)

        # optimize both critics
        self.q_optim.zero_grad()
        critic_loss1.backward()
        self.q_optim.step()

        self.q2_optim.zero_grad()
        critic_loss2.backward()
        self.q2_optim.step()

        # delay policy update
        if self.update_counter % self.policy_delay == 0:
            # compute actor loss
            #print('updating policy, update_counter:', self.update_counter)
            actor_loss = -self.q(state, self.pi(state)).mean()

            # optimize actor
            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()

            # update target networks
            cu.soft_update_params(self.q, self.q_target, self.tau)
            cu.soft_update_params(self.q2, self.q2_target, self.tau)
            cu.soft_update_params(self.pi, self.pi_target, self.tau)

        return {}