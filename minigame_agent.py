import random
import math
import numpy as np
import pandas as pd
from collections import deque

import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from modified_state_space import state_modifier.modified_state_space

LAST_TIMESTEP = environment.StepType.LAST # MEANS ITS DONE

_NO_OP = actions.FUNCTIONS.no_op.id
_ATTACK = 1 #TODO
_RETREAT = 2 #TODO

# _SELECT_POINT = actions.FUNCTIONS.select_point.id
# _SELECT_ARMY = actions.FUNCTIONS.select_army.id
# _ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_PLAYER_SELF = 1

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

SQUARE_INPUT = tf.placeholder(tf.float32, [ , , 3] ) # SQUARE INPUTS
SCALAR_INPUT = tf.placeholder(tf.float, [, ] )

network_action = tf.placeholder(tf.float32, [None, len(ACTIONS)] ) # target
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


def conv_net(square_input, scalar_input, weights, biases, dropout):
    """
        square_input:
            3D input ( x x )
        scalar_input:
            1D input ( )
        weights:
            scalar:
            wc1:
            wc2:
            wd1:

        biases:
            scalar
            bc1
            bc2
            bd1
    """
    square_input = tf.reshape(square_input, shape=[-1, 28, 28, 1])
    scalar = tf.reshape(scalar_input, shape=[-1, 100]) # TODO

    scalar_layer = tf.matmul(scalar, weights['scalar'], biases['scalar'])

    # Convolution Layer
    conv1 = conv2d(square_input, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    conv_out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    out = tf.add(conv_out + scalar_out)
    return out




ACTIONS = [_NO_OP, _ATTACK, _RETREAT]

INPUT_DIMENSION = 100 #TODO 

class DefeatRoachesAgent(base_agent.BaseAgent):
    def __init__(self):
        self.memory = deque(maxlen=10000)
        self.games_elapsed = 0

        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01
        self.alpha_decay = 0.01
        self.batch_size = 64

        self.frame = 0
        self.FRAMES_PER_ACTION = 10

        self.state = None
        self.action = None
        self.reward = 0

        # initialize model
        self.model = self.initialize_model()
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay) )

        super(SmartAgent, self).__init__()

    def initialize_model(self):
        square_input, scalar_input = get_input_space()
        target = get_action_space()
        
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, num_classes])),
            'scalar':tf.Variable(tf.random_normal([7*7*64, 1024])) # TODO
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_classes])),
            'scalar': tf.Variable(tf.random_normal([num_classes])) #TODO
        }
        
        return conv_net(square_input, scalar_input, weights, biases, dropout)

    def remember(self, state, action, reward, next_state, done):
        '''
        Store state-action tuple to memory
        '''
        self.memory.append( (state, action, reward, next_state, done) )

    def choose_action(self, state, epsilon):
        '''
        Choose action using epsilon greedy approach
        Take random action in action space if random() < epsilon
        Otherwise, take action that maximizes Q(s, a) value
        '''
        if np.random.random() < epsilon:
            return (int) (np.random.random() * len(ACTIONS) )
        else:
            return np.argmax( self.model.predict(state) )

    def get_epsilon(self, t):
        '''
        Gets the epsilon value used in epsilon greedy approach
        epsilon decays as t increases
        '''

        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10( (t+1) * self.epsilon_decay ) ) )

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size)) # sample at most batch_size elements from memory

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict( state )
            if done: # value is reward if done
                y_target[0][action] = reward
            else: # otherwise q value is reward + gamma * Qt+1 (bellman equation)
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])

            x_batch.append( state[0] )
            y_batch.append( y_target[0] )

        # put batch in for training
        self.model.fit( np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # decay


    # def transformLocation(self, x, x_distance, y, y_distance):
    #     if not self.base_top_left:
    #         return [x - x_distance, y - y_distance]
        
    #     return [x + x_distance, y + y_distance]
    
    # def step(self, obs):
    #     super(SmartAgent, self).step(obs)
        
    #     player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    #     self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
    #     return actions.FunctionCall(_NO_OP, [])

    def handle_step(self, next_state, reward, done):
        if done:
            self.compute_loss()
            return ACTION_NO_OP

        if frame == 0: # selects the random unit
            return actuator.select_random_unit()

        elif frame == 1: # actually selects the action
            chosen_action = self.choose_action(state, self.epsilon)


            return chosen_action, previous_reward, previous_action
        else: # skips frame
            frame = (frame + 1) % self.FRAMES_PER_ACTION
            return ACTION_NO_OP

    def step(self, observation):
        next_state, reward, done = self.extract_from_observation(observation)
        action = self.handle_step(next_state, reward, done)
        if action not in ALLOWED_ACTIONS:
           return action

        # calculate reward, etc..
        prev_reward = self.reward
        prev_state = self.state
        prev_action = self.action

        # select OUR ACTIONS... retreat, attack closest, attack weakest       
       

    def extract_done_from_observation(self, observation):
        return observation.step_type == LAST_TIMESTEP

    def extract_reward_from_observation(self, observation):
        '''
            Returns the reward by taking the delta in score
        '''
        curr_score = observation.observation['score_cumulative'][0]
        return curr_score - self.score

    def extract_state_from_observation(self, observation):
        return modified_state_space(observation)

    def extract_from_observation(self, observation):
        done = extract_done_from_observation(self, observation)
        reward = extract_reward_from_observation(self, observation)
        state = extract_state_from_observation(observation)

        return state, reward, done

    def preprocess_state(self, state):
        '''
        Change state into INPUT SPACE FOR DQN
        '''
        # TODO
        return np.reshape(state, [1,4])