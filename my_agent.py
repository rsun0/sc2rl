# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=300)

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import environment
#from pysc2.customlib.modified_state_space import state_modifier

# END PYSC2 IMPORTS
# BEGIN PYTORCH IMPORTS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# END PYTORCH IMPORTS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_HITPOINTS = features.SCREEN_FEATURES.unit_hit_points.index
_PLAYER_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_PLAYER_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
_ARMY_COUNT = 8
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_FIRST_TIMESTEP = environment.StepType.FIRST
_LAST_TIMESTEP = environment.StepType.LAST


DEPTH = 3
HEIGHT = 84
WIDTH = 84



class SingleUnitAction(nn.Module):
    
    def __init__(self):
        super(SingleUnitAction, self).__init__()
        self.activation = F.relu
        
        self.input_shape = (1, DEPTH, HEIGHT, WIDTH)
        
        self.conv1 = nn.Conv3d(in_channels = 1,
                                    out_channels=10,
                                    kernel_size=(DEPTH, 10, 10),
                                    padding = (0, 1, 1))
                                    
        self.n_size = self._get_conv_output(self.input_shape)
        self.fc1_size = 600
        self.fc2_size = 200
        self.conv1.double()
        
        self.fc1 = nn.Linear(self.n_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, 3)
        #self.fc3 = nn.Linear(self.fc2_size, 3)
        
        self.fc1.double()
        self.fc2.double()
        #self.fc3.double()
        
        self.loss = nn.MSELoss()
        
    def _get_conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        output_feat = self._forward_features(inp)
        n_size = output_feat.data.view(output_feat.size(0), -1)
        return n_size.size()[1]
        
    def _forward_features(self, x):
        x = F.max_pool3d(self.activation(self.conv1(x)), (1, 2, 2))
        return x
        
    def forward(self, x):
        
        x = self._forward_features(x)
        x = x.view(-1, self.n_size)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        #x = self.fc3(x)
        return x



class DefeatRoaches(base_agent.BaseAgent):

  def __init__(self):
    print("The agent is being constructed again.")
    self.policy = SingleUnitAction()
    self.replay_memory = []
    self.current_game = []
    self.current_sarsd = []
    self.previous_score = 0
    self.num_steps = 0
    self.epsilon_max = 1
    self.epsilon_decay = 10000.0
    self.start_training = 500
    self.optimizer = optim.SGD(self.policy.parameters(), lr=3e-5, momentum=0.0)
    
    self.errors = []
    self.previous_action = 0
    self.frames_between_actions = 10
    self.training = True
    self.game_ended = False

  """An agent specifically for solving the DefeatRoaches map."""
  def step(self, obs):
    self.num_steps += 1    

    super(DefeatRoaches, self).step(obs)
    
    steptype = obs.step_type
    if (steptype == _LAST_TIMESTEP):
        self.game_ended = True
        """if (len(self.current_sarsd) == 2):
            x = obs.observation["screen"][_PLAYER_HOSTILE]
            self.current_sarsd.append(self.compute_end_game_reward(x))
            for i in x:
                print(i)
                """
        self.replay_memory.append(self.current_game)
        self.current_game = []
        print("GAME HAS ENDED")
        
    
    if (self.num_steps % self.frames_between_actions != 0):
      #player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      #if (has_restarted(player_relative, self.current_sars))
      return actions.FunctionCall(_NO_OP, [])
      
    print ("Step " + str(self.num_steps))
    if _ATTACK_SCREEN in obs.observation["available_actions"]:
    
      #TEST = state_modifier.modified_state_space(obs)
    
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      player_friendly = (player_relative == _PLAYER_FRIENDLY).astype(int)
      player_hostile = (player_relative == _PLAYER_HOSTILE).astype(int)  #/ float(_PLAYER_HOSTILE)
      player_hit_points = obs.observation["screen"][_PLAYER_HITPOINTS]
      #for i in range(len(player_hit_points)):
      #  print(player_hit_points[i])
      #x = obs.observation["minimap"]
      #y = obs.observation["game_loop"]
      #print(obs.observation.keys())
      selected_units = obs.observation["multi_select"]
      #print(selected_units)
      roach_y, roach_x = (player_relative == _PLAYER_HOSTILE).nonzero()
      marine_y, marine_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not roach_y.any():
        return actions.FunctionCall(_NO_OP, [])
      
      ### COMPUTE IDEAL ACTION
      
      for i in player_hostile:
        print (i)
      net_input = np.stack([player_friendly, player_hostile, player_hit_points], axis=0)
      action = self.action(net_input)
      score = obs.observation["score_cumulative"][0]
      steptype = obs.step_type
      done = False
      if (self.game_ended):
        self.game_ended = False
        reward = self.compute_end_game_reward(net_input)
        done = True
      else:
        #reward = self.compute_reward(net_input)
        reward = (score - self.previous_score) / 5
        self.previous_score = score
        if (self.previous_action == 2):
          reward -= 0.08
      self.previous_score = score
      ### STORE REPLAY MEMORY
      if (len(self.current_sarsd) > 0):
        self.current_sarsd.append(reward / 10)
        self.current_sarsd.append(net_input)
        self.current_sarsd.append(done)
        self.current_game.append(self.current_sarsd)
      self.current_sarsd = [net_input]
      self.current_sarsd.append(action)
      if (self.num_steps > self.start_training and self.training):
        e = self.gradient_step(80)
        self.errors.append(e)
        
      if (len(self.errors) % 50 == 0):
        self.plot_errors()
      
      ###  HANDLE RANDOM BEHAVIOR
      if (self.num_steps < self.start_training):
        action = random.randint(0, 2)
      elif (random.random() < math.e ** (-1 * (self.num_steps - self.start_training) / self.epsilon_decay)):
        action = random.randint(0, 2)
      
      self.previous_action = action
      if (action == 0):
        closest_index = self.compute_closest(roach_x, roach_y, marine_x, marine_y)
        target = [closest_index[0], closest_index[1]]
        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        
      if (action == 1):
        weakest_index = self.compute_weakest(roach_x, roach_y, player_hit_points)
        target = [weakest_index[0], weakest_index[1]]
        return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        
      if (action == 2):
        retreat_index = self.compute_retreat(roach_x, roach_y, marine_x, marine_y)
        target = [retreat_index[0], retreat_index[1]]
        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    
        
    elif _SELECT_ARMY in obs.observation["available_actions"]:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
    else:
      return actions.FunctionCall(_NO_OP, [])
 
  def action(self, net_input):
    net_input = net_input.reshape(1, 1, DEPTH, HEIGHT, WIDTH)
    net_input = Variable(torch.from_numpy(net_input))
    net_input = net_input.type(torch.DoubleTensor)
    output = self.policy(net_input).data.numpy()
    return np.argmax(output)
    
  def compute_reward(self, net_input):
    if (self.num_steps <= 1):
        return 0
    
    past_input = self.current_sarsd[0]
    enemy_unit_reward = np.multiply(past_input[1], past_input[2]) - np.multiply(net_input[1], net_input[2])
    friendly_unit_reward = np.multiply(net_input[0], net_input[2]) - np.multiply(past_input[0], net_input[2])
    enemy_sum = np.sum(enemy_unit_reward)
    friendly_sum = np.sum(friendly_unit_reward)
    #print(enemy_sum, friendly_sum)
    return (friendly_sum + enemy_sum) / 5000
    
    
    
  def compute_end_game_reward(self, net_input):
    enemies = net_input[1]
    s = np.sum(enemies)
    print(s)
    if (s > 0):
        return -10
    else:
        return 10
    
  def plot_errors(self):
    plt.figure(1)
    plt.clf()
    errors = np.array(self.errors)
    plt.title('Training . . .')
    plt.xlabel('Episode')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.pause(0.001)
 
  def gradient_step(self, batch_size):
    states = []
    actions = []
    rewards = []
    next_states = []
    length = len(self.replay_memory)    
    for i in range(batch_size):
        index = random.randint(0, length-1)
        length2 = len(self.replay_memory[index])
        while (len(self.replay_memory[index]) == 0):
            index = random.randint(0, length-1)
            length2 = len(self.replay_memory[index])
        index2 = random.randint(0, length2-1)
        states.append(self.replay_memory[index][index2][0])
        actions.append(self.replay_memory[index][index2][1])
        rewards.append(self.replay_memory[index][index2][2])
        next_states.append(self.replay_memory[index][index2][3])
    states = np.array(states)
    actions = self.action_mask(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    
    s = states.shape
    states = states.reshape((s[0], 1, s[1], s[2], s[3])) 
    next_states = next_states.reshape((s[0], 1, s[1], s[2], s[3])) 
    
    states = Variable(torch.from_numpy(states))
    states = states.type(torch.DoubleTensor)
    next_states = Variable(torch.from_numpy(next_states))
    next_states = next_states.type(torch.DoubleTensor)
    
    targets = self.compute_targets(rewards, next_states)
    targets = Variable(torch.from_numpy(targets))
    targets = targets.type(torch.DoubleTensor)
    
    output = self.policy(states)
    output = torch.masked_select(output, actions)
    self.optimizer.zero_grad()
    error = self.policy.loss(output, targets)
    error.backward()
    self.optimizer.step()
    print("The error is " + str(error.data.numpy()[0]))
    return error.data.numpy()[0]
    
    
  def action_mask(self, actions):
    actions = np.array(actions)
    mask = np.array([[1 if actions[j] == i else 0 for i in range(3)] for j in range(len(actions))])
    mask = Variable(torch.from_numpy(mask)).type(torch.ByteTensor)
    return mask
    
  def compute_targets(self, rewards, next_states):
    gamma = 0.99
    value_of_next_states = self.policy(next_states).data.numpy()
    value_of_next_states = np.amax(value_of_next_states, axis=1)
    Q = rewards + gamma * value_of_next_states
    return Q
  
  
  
      
  def compute_closest(self, roach_x, roach_y, marine_x, marine_y):
    closest = [roach_y[0], roach_x[0]]
    distances = np.array([], dtype=float)
    for j in range(len(roach_x)):
        dist = 0.0
        for i in range(len(marine_x)):
            dist += ( (roach_x[j] - marine_x[i]) ** 2 + (roach_y[j] - marine_y[i]) ** 2)
        distances = np.append(distances, [dist])
    index = np.argmin(distances)
    return np.array([roach_x[index], roach_y[index]])
    
  def compute_weakest(self, roach_x, roach_y, player_hit_points):
    weakest = [roach_x[0], roach_y[0]]
    lowest_health = player_hit_points[roach_y[0]][roach_x[0]]
    for i in range(len(roach_x)):
        x = roach_x[i]
        y = roach_y[i]
        if (player_hit_points[y][x] < lowest_health):
            lowest_health = player_hit_points[y][x]
            weakest = [x, y]
    return np.array(weakest)
    
  def compute_retreat(self, roach_x, roach_y, marine_x, marine_y):
    avg_roach_x = np.mean(roach_x)
    avg_roach_y = np.mean(roach_y)
    avg_marine_x = np.mean(marine_x)
    avg_marine_y = np.mean(marine_y)
    retreat_vector = np.array([avg_roach_x - avg_marine_x, avg_roach_y - avg_marine_y])
    target_x = avg_marine_x
    target_y = avg_marine_y
    while (0 < target_x < WIDTH and 0 < target_y < HEIGHT):
        target_x -= 0.5 * retreat_vector[0]
        target_y -= 0.5 * retreat_vector[1]
    target_x += 0.5 * retreat_vector[0]
    target_y += 0.5 * retreat_vector[1]
    return [target_x, target_y]  
    
  def has_restarted(self):
    return;
      
      
      
      
      
