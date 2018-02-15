import random
import math
import numpy as np
import pandas as pd
from collections import deque

import tensorflow as tf

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

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

class SmartAgent(base_agent.BaseAgent):
    def __init__(self, n_episodes=10000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet

        # initialize model
        


        # self.model = Sequential()
        # self.model.add( Convolution2D(32, 3, 3, activation='relu', input_shape=(128,128,3)) ) # input layer
        # self.model.add(MaxPooling2D(pool_size=(2,2)))
        # self.model.add(Dropout(0.25))
        # self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(len(ACTIONS), activation='softmax')) # output layer (actions space)

        # Store layers weight & bias
        
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

        self.model = conv_net(square_input, scalar_input, weights, biases, dropout)


        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay) )

        super(SmartAgent, self).__init__()

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

    def preprocess_state(self, state):
        '''
        Change state into INPUT SPACE FOR DQN
        '''
        # TODO
        return np.reshape(state, [1,4])

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
    
    def step(self, obs):
        super(SmartAgent, self).step(obs)
        
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        return actions.FunctionCall(_NO_OP, [])

    
