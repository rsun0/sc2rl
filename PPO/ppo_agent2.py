# OBJECTIVE: maximize theta in SUM (n 1->N) ( pi_theta(an | sn) / pi_theta_old(an | sn) * ADVn ) - C * KL(pi_theta, pi_theta_old)

# for i = 1,2, ...
#   run pi_theta for T timesteps
#   estimate adv function for all timesteps using NN
#   do SGD on objective
#   (consequence: if KL too high, increasing B. if KL too low, decrease B)

# based on code from https://github.com/wooridle/DeepRL-PPO-tutorial/blob/master/ppo.py

import tensorflow as tf
import numpy as np
from custom_env import MinigameEnvironment
from modified_state_space import state_modifier
import random
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
import action_interface

np.set_printoptions(linewidth=200, precision=4)

class Network(object):
    def __init__(self, env, scope, num_layers, num_units, obs_plc, act_plc, select_act_plc, tl_plc, trainable=True):
        
        self.filters1 = 16
        self.filters2 = 32
        self.filters3 = 64
    
    
        self.env = env
        self.observation_size = obs_plc
        self.action_size = env.action_space
        self.select_size = env.select_space
        self.select_width = env.select_space
        self.select_height = env.select_space
        self.trainable = trainable
        self.activation = tf.nn.leaky_relu
        self.scope = scope

        self.obs_place = obs_plc
        self.acts_place = act_plc
        self.select_acts_place = select_act_plc
        self.tl_plc = tl_plc

        self.p , self.v, self.select_p, self.logstd, self.select_logstd = self._build_network(num_layers=num_layers, num_units=num_units)
        self.act_op = self.action_sample()
        
        

    def _build_network(self, num_layers, num_units):
        with tf.variable_scope(self.scope):
            x = self.obs_place
            
            # Initializes convolutional layers
            x = tf.layers.conv2d(x,
                filters=self.filters1,
                kernel_size=[8, 8],
                padding="same",
                strides=(4, 4),
                activation=self.activation)

            baseline_conv_output = tf.layers.conv2d(x,
                filters=self.filters2,
                kernel_size=[4, 4],
                padding="same",
                strides=2,
                activation=self.activation)

            x = tf.contrib.layers.flatten(baseline_conv_output)
            
            # Initializes fully connected layers
            for i in range(num_layers):
                fc_mul = 1 #num_layers - i
                x = tf.layers.dense(x, 
                                units=(fc_mul * num_units), 
                                activation=self.activation, 
                                name="p_fc"+str(i), 
                                trainable=self.trainable)
            
            
            
            action = tf.layers.dense(x, 
                                units=self.action_size, 
                                activation=tf.nn.softmax,
                                name="p_fc"+str(num_layers), 
                                trainable=self.trainable)
                
                
            select_p_x1 = tf.layers.dense(x, units=self.select_width, activation=tf.nn.softmax, name="select_p_x1_fc", trainable=self.trainable)
            select_p_y1 = tf.layers.dense(x, units=self.select_height, activation=tf.nn.softmax, name="select_p_y1_fc", trainable=self.trainable)


            ### FC layers for bot right
            #select_p_br = tf.concat([x, self.tl_plc], axis=-1)
            
            ### Placeholders for bot right         
            select_p_x2 = tf.layers.dense(x, units=self.select_width, activation=tf.nn.softmax, name="select_p_x2_fc", trainable=self.trainable)
            select_p_y2 = tf.layers.dense(x, units=self.select_height, activation=tf.nn.softmax, name="select_p_y2_fc", trainable=self.trainable)
            
            
            
            value = tf.layers.dense(x, units=1, activation=None, name="v_fc"+str(num_layers), trainable=self.trainable)

            
            
            

            logstd = tf.get_variable(name="logstd", shape=[self.action_size],
                                     initializer=tf.zeros_initializer)
                                     
            select_logstd = tf.get_variable(name="select_logstd", shape=[self.select_size], initializer=tf.zeros_initializer)

        return action, value, [select_p_x1, select_p_y1, select_p_x2, select_p_y2], logstd, select_logstd

    def action_sample(self):
        return self.p #+ tf.exp(self.logstd) * tf.random_normal(tf.shape(self.p))

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOAgent(object):
    def __init__(self, env, session=None):
        self.env = env

        self.input_shape = self.env.observation_space
        self.session=session
        
        
        ### hyperparameters - TODO: TUNE
        self.learning_rate = 1e-4
        
        ### weight for vf_loss
        self.c1 = 1
        
        ### weight for entropy
        self.c2 = 0
        
        ### Constant used for numerical stability in log and division operations
        self.epsilon = 1e-8
        
        self.epochs = 3
        self.step_size = 1024
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_param = 0.1
        self.batch_size = 32
        self.hidden_size = 256
        self.averages = []

        ## placeholders
        self.adv_place = tf.placeholder(shape=[None], dtype=tf.float32)
        self.return_place = tf.placeholder(shape=[None], dtype=tf.float32)

        self.obs_place = tf.placeholder(shape=([None] + env.observation_space),
                                        name="ob", dtype=tf.float32)
        self.acts_place = tf.placeholder(shape=(None,self.env.action_space),
                                         name="ac", dtype=tf.float32)
                                         
        self.select_acts_place = tf.placeholder(shape=(None, 4, self.env.select_space),
                                         name="sac", dtype=tf.float32)

        self.tl_place = tf.placeholder(shape=[None, 2*env.select_space], dtype=tf.float32)

        ## build network
        self.net = Network(env=self.env,
                           scope="pi",
                           num_layers=1,
                           num_units=self.hidden_size,
                           obs_plc=self.obs_place,
                           act_plc=self.acts_place,
                           tl_plc = self.tl_place,
                           select_act_plc=self.select_acts_place)

        self.old_net = Network(env=self.env,
                               scope="old_pi",
                               num_layers=1,
                               num_units=self.hidden_size,
                               obs_plc=self.obs_place,
                               act_plc=self.acts_place,
                               tl_plc=self.tl_place,
                               select_act_plc=self.select_acts_place,
                               trainable=False)

        # tensorflow operators
        self.assign_op = self.assign(self.net, self.old_net)
        self.select_ent, self.select_pol_loss, self.vf_loss, self.select_update_op = self.select_update()
        self.move_ent, self.move_pol_loss, self.vf_loss, self.move_update_op = self.move_update()
        self.saver = tf.train.Saver()

    def select_logp(self, net):
        logp = 0
        for i in range(4):
            p = net.select_p[i]
            logp += (-(0.5 * tf.reduce_sum(tf.square((net.select_acts_place[:,i] - p) / tf.exp(net.select_logstd)), axis=-1) \
                + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(p)[-1]) \
                + tf.reduce_sum(net.select_logstd, axis=-1)) )
        return logp / 4
        
        
    def move_logp(self, net):
        logp = -(0.5 * tf.reduce_sum(tf.square((net.acts_place - net.p) / tf.exp(net.logstd)), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(net.p)[-1]) \
            + tf.reduce_sum(net.logstd, axis=-1))

        return logp

    def move_entropy(self, net, batch_size):
        #ent = tf.reduce_sum(net.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        ent = tf.reduce_sum(net.p * tf.log(net.p + self.epsilon))
        return - (ent / batch_size)

    def select_entropy(self, net, batch_size):
        #ent = tf.reduce_sum(net.select_logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        ent = tf.reduce_mean(net.select_p[0] * tf.log(net.select_p[0] + self.epsilon))
        for i in range(3):
            p = net.select_p[i+1]
            ent += tf.reduce_sum(p * tf.log(p + self.epsilon))
        return -(ent / batch_size)

    def assign(self, net, old_net):
        assign_op = []
        for (newv, oldv) in zip(net.get_variables(), old_net.get_variables()):
            assign_op.append(tf.assign(oldv, newv))

        return assign_op

    
    """
        Returns 2-phase step structure. Each step consists of
            1. Selection
            2. Movement
        Selection chooses a [x1,y1,x2,y2] box over which to select units.
        Movement takes the selected units and performs actions as per action_interface.
        
        Value, reward for each step is the sum of these quantities for both steps.
        
    """   
    
    def traj_generator(self):
        t = 0
        
        done = True
        ob, reward, done, _ = env.reset()
        ob = self.state_reshape(ob)
        ob = self.normalize(ob)
        cur_ep_return = 0
        cur_ep_length = 0
        
        scores = []
        num_games = 0
        
        #obs = np.array([ob, ob]for _ in range(self.step_size)]).T
        #select_obs = np.array([ob for _ in range(self.step_size)])
        #move_obs = np.array([ob for _ in range(self.step_size)])
        select_obs = np.zeros((self.step_size, ob.shape[0], ob.shape[1], ob.shape[2]), 'uint8')
        move_obs = np.zeros((self.step_size, ob.shape[0], ob.shape[1], ob.shape[2]), 'uint8')
        rewards = np.zeros(self.step_size, 'float32')
        values = np.zeros(self.step_size, 'float32')
        dones = np.zeros(self.step_size, 'int32')
        select_actions = np.zeros((self.step_size, 4, self.env.select_space), 'float32')
        move_actions = np.zeros((self.step_size, self.env.action_space), 'float32')
        
        prev_select_actions = select_actions.copy()
        prev_move_actions = move_actions.copy()
        
        # Records selected topleft coordinates
        tl_plc_in = np.zeros((self.step_size, 2*self.env.select_space), 'float32')
        
        
        selection = 0
        movement = 0
        prev_selection = [-1 for i in range(4)]
        prev_movement = -1
        
        ep_returns = []
        ep_lengths = []
        value = 0
        
        while True:
        
            ### Stores index of current step
            i = int((t % (2 * self.step_size)) / 2)
        
            ### Even if selecting, odd if moving
            j = t % 2
        
            ### Handles return ###
            if (t > 0 and (t % (2*self.step_size)) == 0):
                self.averages.append(sum(scores) / (1+num_games))
                print("Average game score of this batch: {}".format(self.averages[-1]))
                scores = []
                num_games = 0
                
                yield {"select_ob": select_obs, 
                       "move_ob": move_obs, 
                       "reward":rewards, 
                       "value": values,
                       "done": dones, 
                       "select_action": select_actions,
                       "move_action": move_actions,
                       "prevselect_action": prev_select_actions,
                       "prevmove_action": prev_move_actions,
                       "tl_plc_in": tl_plc_in,
                       "nextvalue": value*(1-done), 
                       "ep_returns": ep_returns,
                       "ep_lengths": ep_lengths
                       }

                ep_returns = []
                ep_lengths = []
                
                rewards = np.zeros(self.step_size, 'float32')
                values = np.zeros(self.step_size, 'float32')
                
                
            ### Handles selection ###
            
            if (j == 0):
                prev_selection = selection
                selection_nums, value = self.select(self.normalize(ob))
                #selection_nums = self.select_selection(selection_vals)
                
                select_obs[i] = ob
                values[i] += value
                dones[i] = done
                
                
                transformed_select = selection_nums
                transformed_select[2:] -= transformed_select[:2]
                if (transformed_select[2] == self.env.select_space - 1):
                    transformed_select[2] = random.randint(transformed_select[2], self.env.select_space-1)
                if (transformed_select[3] == self.env.select_space - 1):
                    transformed_select[3] = random.randint(transformed_select[3], self.env.select_space-1)
                
                select_actions[i] = np.zeros((4, self.env.select_space,))
                select_actions[i, range(4), transformed_select] = 1
                prev_select_actions[i] = np.zeros((4, self.env.select_space,))
                prev_select_actions[i, range(4), prev_selection] = 1
                tl_plc_in[i] = np.zeros((2*self.env.select_space,))
                tl_plc_in[i][selection_nums[0]] = 1
                tl_plc_in[i][self.env.select_space + selection_nums[1]] = 1
                #csa = converted_select_action = self.select_convert(selection_nums)
                ob, temp_reward, done, _ = self.env.step(0, topleft=selection_nums[:2], botright=selection_nums[2:])
                
                ob = self.state_reshape(ob)
                reward += temp_reward
                
                cur_ep_return += reward
                cur_ep_length += 1
                
                ### Timestep finished on select move;
                ### Add reward, value to previous timestep reward, value
                if done:
                    
                    ### Cancel in case this is first step
                    if (i == 0):
                        ob, reward, done, _ = env.reset()
                        ob = self.state_reshape(ob)
                        t += 1
                        continue
                        
                    t -= 1
                    i -= 1
                    dones[i] = done
                    values[i] = (2*values[i] + value) / 3
                    rewards[i] += reward
                    scores.append(cur_ep_return)
                    num_games += 1
                    print("Reward: {}. {} of {} steps.".format(cur_ep_return, t%(2*self.step_size), (2*self.step_size)))
                    ep_returns.append(cur_ep_return)
                    ep_lengths.append(cur_ep_length)
                    cur_ep_return = 0
                    cur_ep_length = 0
                    ob, reward, done, _ = env.reset()
                    ob = self.state_reshape(ob)
                    t += 1
                    i += 1
                    continue
                    
                selection = transformed_select
                    
                    
            ### Handles movement ###
            
            else:
                prev_movement = movement
                action_vals, value = self.act(self.normalize(ob))
                movement = self.select_action(action_vals)
                move_obs[i] = ob
                values[i] = (values[i] + value) / 2
                dones[i] = done
                move_actions[i] = np.zeros((self.env.action_space,))
                move_actions[i][movement] = 1
                prev_move_actions[i] = np.zeros((self.env.action_space,))
                prev_move_actions[i][prev_movement] = 1
                
                ob, temp_reward, done, _ = self.env.step(movement)
                ob = self.state_reshape(ob)
                ob = self.normalize(ob)
                reward += temp_reward
                rewards[i] = reward
                
                cur_ep_return += temp_reward
                cur_ep_length += 1
                
                if done:
                    scores.append(cur_ep_return)
                    num_games += 1
                    print("Reward: {}. {} of {} steps.".format(cur_ep_return, t%(2*self.step_size), (2*self.step_size)))
                    ep_returns.append(cur_ep_return)
                    ep_lengths.append(cur_ep_length)
                    cur_ep_return = 0
                    cur_ep_length = 0
                    ob, reward, done, _ = env.reset()
                    ob = self.state_reshape(ob)
                    ob = self.normalize(ob)
                
            ### Handles end of 2-phase step        
            if (j == 1):
                reward = 0
                value = 0
                
            ### Increments time count    
            t += 1

    def act(self, ob):
        actions, value = self.session.run([self.net.act_op, self.net.v], feed_dict={
            self.net.obs_place: ob[None]
        })
        return actions, value
        
    def select(self, ob):
        x1, y1, value = self.session.run(self.net.select_p[:2] + [self.net.v], feed_dict={ self.net.obs_place: ob[None]})
        tl = self.select_selection(np.array([x1[0], y1[0]]))
        tl_plc = np.zeros((2*self.env.select_space))
        x1 = tl[0]
        y1 = tl[1]
        
        tl_plc[x1] = 1
        tl_plc[self.env.select_space+y1] = 1
        x2, y2 = self.session.run(self.net.select_p[2:], feed_dict={ self.net.obs_place: ob[None], self.net.tl_plc: tl_plc[None]
        })
        br = self.select_selection(np.array([x2[0], y2[0]]))
        
        x2 = min((x1+br[0]), self.env.select_space-1)
        y2 = min((y1+br[1]), self.env.select_space-1)
        
        
        
        return np.array([x1, y1, x2, y2]), value
        
    def normalize(self, ob):
        #return ((ob.T - np.mean(ob, axis=(1,2))) / (1 + np.max(ob, axis=(1,2)) - np.min(ob, axis=(1,2)))).T
        x, y, z = ob.shape
        ob = ob.reshape((1, x, y, z))
        self.state_normalize(ob)
        ob = ob.reshape((x, y, z))
        return ob
        
    def state_normalize(self, ob):
        self.zero_one_norm(ob[:,:,:,1])
        self.zero_one_norm(ob[:,:,:,3])
        
        
    def zero_one_norm(self, array):
        n = array.shape[0]
        arr_max = np.max(array, tuple(range(1, len(array.shape)))).reshape((n, 1, 1))
        arr_min = np.min(array, tuple(range(1, len(array.shape)))).reshape((n, 1, 1))
        denom = arr_max - arr_min + 1
        array -= arr_min
        array = array *  (1/denom)
        
    """
    Returns some element i in range(len(action_probs)), each with action_probs[i] probability.
    """
    def select_action(self, action_probs):
        action_probs = action_probs.reshape((self.env.action_space))
        num = random.random()
        running_sum = 0.0
        for i in range(len(action_probs)):
            
            running_sum += action_probs[i]
            if num < running_sum:
                return i
            
            
        return len(action_probs)-1
        
    """
        select_selection takes in selection_probs: numpy array of shape (4, self.env.select_space)
        returns: list of length 4
    """
    def select_selection(self, selection_probs):
        output = []
        #selection_probs = selection_probs.reshape((4, self.env.select_space))
        for i in range(selection_probs.shape[0]):
            num = random.random()
            running_sum = 0.0
            for j in range(len(selection_probs[i])):
                running_sum += selection_probs[i][j]
                if num < running_sum:
                    output.append(j)
                    break
        return output
        
        
    """
        Augments data by generating the 8 equivalent transformations of states and actions. See: dihedral group
        
    """
    def rotateReflectAugmentation(self, traj):
        ''' 
        select_obs shape: (self.step_size, 84, 84, 9)
        {"select_ob": select_obs, 
                       "move_ob": move_obs, 
                       "reward":rewards, 
                       "value": values,
                       "done": dones, 
                       "select_action": select_actions,
                       "move_action": move_actions,
                       "prevselect_action": prev_select_actions,
                       "prevmove_action": prev_move_actions,
                       "tl_plc_in": tl_plc_in,
                       "nextvalue": value*(1-done), 
                       "ep_returns": ep_returns,
                       "ep_lengths": ep_lengths
                       }
        '''
        
        super_traj = {}
        super_traj["reward"] = np.tile(traj["reward"], 8)
        super_traj["value"] = np.tile(traj["value"], 8)
        super_traj["done"] = np.tile(traj["done"], 8)
        super_traj["nextvalue"] = np.tile(traj["nextvalue"], 8)
        super_traj["ep_returns"] = np.tile(traj["ep_returns"], 8)
        super_traj["ep_lengths"] = np.tile(traj["ep_lengths"], 8)
        super_traj["advantage"] = np.tile(traj["advantage"], 8)
        super_traj["return"] = np.tile(traj["return"], 8)
        
        ### Initial values. They will be appended to.
        super_traj["select_ob"] = traj["select_ob"]
        super_traj["move_ob"] = traj["move_ob"]
        super_traj["select_action"] = traj["select_action"]
        super_traj["move_action"] = traj["move_action"]
        super_traj["tl_plc_in"] = traj["tl_plc_in"]
        
        ### Transformation T to make for easier processing
        select_ob = np.swapaxes( (np.swapaxes( traj["select_ob"], 0, 1)), 1, 2 )
        move_ob = np.swapaxes( (np.swapaxes( traj["move_ob"], 0, 1)), 1, 2 )
        
        select_action = traj["select_action"]
        move_action = traj["move_action"]
        select_action_rot = traj["select_action"]
        move_action_rot = traj["move_action"]
        
        select_ob_rot = select_ob
        move_ob_rot = move_ob
        
        ### Reflections and Rotations
        for i in range(4):
        
            ### Reflection of observations
            select_ob_ref = np.flip(select_ob_rot, 0)
            move_ob_ref = np.flip(move_ob_rot, 0)
            
            select_action_ref = self.select_ref(select_action)
            move_action_ref = self.move_ref(move_action)
            tl_plc_in_ref = np.zeros((self.step_size, 2*self.env.select_space,))
            tl_plc_in_ref[:,:self.env.select_space] = select_action_ref[:,0,:]
            tl_plc_in_ref[:,self.env.select_space:] = select_action_ref[:,1,:]
            
            ### Inverse transformation
            select_ob_ref = np.swapaxes( (np.swapaxes( select_ob_ref, 1, 2)), 0, 1)
            move_ob_ref = np.swapaxes( (np.swapaxes( move_ob_ref, 1, 2)), 0, 1)
            
            super_traj["select_ob"] = np.concatenate([super_traj["select_ob"], select_ob_ref], 0)
            super_traj["move_ob"] = np.concatenate([super_traj["move_ob"], move_ob_ref], 0)
            super_traj["select_action"] = np.concatenate([super_traj["select_action"], select_action_ref], 0)
            super_traj["move_action"] = np.concatenate([super_traj["move_action"], move_action_ref], 0)
            super_traj["tl_plc_in"] = np.concatenate([super_traj["tl_plc_in"], tl_plc_in_ref], 0)
        
            ### Rotation of observations
            
            # No need to perform a rotation of 0
            if (i == 0):
                continue
                
                
            select_ob_rot = np.rot90( select_ob, i )
            move_ob_rot = np.rot90( move_ob, i )
            
            select_action_rot = self.select_rot(select_action_rot, 1)
            move_action_rot = self.move_rot(move_action_rot, 1)
            
            tl_plc_in_rot = np.zeros((self.step_size,2*self.env.select_space))
            tl_plc_in_rot[:,:self.env.select_space] = select_action_rot[:,0,:]
            tl_plc_in_rot[:,self.env.select_space:] = select_action_rot[:,1,:]
            
            ### Inverse transformation
            select_ob_in = np.swapaxes( (np.swapaxes( select_ob_rot, 1, 2)), 0, 1)
            move_ob_in = np.swapaxes( (np.swapaxes( move_ob_rot, 1, 2)), 0, 1)

            super_traj["select_ob"] = np.concatenate([super_traj["select_ob"], select_ob_in], 0)
            super_traj["move_ob"] = np.concatenate([super_traj["move_ob"], move_ob_in], 0)
            super_traj["select_action"] = np.concatenate([super_traj["select_action"], select_action_rot], 0)
            super_traj["move_action"] = np.concatenate([super_traj["move_action"], move_action_rot], 0)
            super_traj["tl_plc_in"] = np.concatenate([super_traj["tl_plc_in"], tl_plc_in_rot], 0)
        
        """
        for i in range(4):
            # rotation
            select_obs_rot = np.rot90( select_obs, i )
            move_ob_rot = np.rot90( move_ob, i )
                    
            # reflection
            select_obs_ref = np.flip(select_obs_rot, (0))
            move_ob_ref = np.flip(move_ob_rot, (0))
            
            select_action_rot = []
            move_action_rot = []
            select_action_ref = []
            move_action_ref = []
            for i in range( self.step_size ):
                s = self.rotate_action( traj["select_action"][i], i)
                m = traj["move_action"][i] # TODO
                
                s_ref = self.flip_action(s)
                m_ref = # TODO
                
                select_action_rot.append(s)
                move_action_rot.append(m)
                
        """
        return super_traj
        ### Transformation T_inv to return back to original format
        #select_obs = np.swapaxes( (np.swapaxes( select_obs, 1, 2)), 0, 1)
        #move_ob = np.swapaxes( (np.swapaxes( move_ob, 1, 2)), 0, 1)
        
    def select_ref(self, selections):
        coords = selections.nonzero()[1].reshape((-1, 2, 2))
        coords[:,0,1] = (self.env.select_space-1) - (coords[:,0,1] + coords[:,1,1])
        coords = coords.reshape((-1, 4))
        selections = (coords[...,None] == np.arange(self.env.select_space)).astype(np.float)
        return selections
    
    def move_ref(self, movements):
        output = np.zeros(movements.shape)
        output[:,0] = movements[:,0]
        output[:,1:4] = movements[:,5:8]
        output[:,4] = movements[:,4]
        output[:,5:8] = movements[:,1:4]
        output[:,9:] = movements[:,9:]
        return output
    
    
    
        
    def select_rot(self, selections, i):
        ### i=1: counter clockwise 90.    i=2: counter clockwise 180.     i=3: counter clockwise 270.
        ### Conversion to 2x2 simple array
        coords = selections.nonzero()[1].reshape((-1, 2, 2))
        for i in range(i):
            coords[0,:] = [coords[0,1], ((self.env.select_space - 1) - (coords[0,0] + coords[1,0]))]
            coords[1,:] = np.flip(coords[1,:])
        coords = coords.reshape((-1, 4))
        output = (coords[...,None] == np.arange(self.env.select_space)).astype(np.float)
        return output
        
        
    def move_rot(self, movements, i):
        ### i=1: counter clockwise 90.    i=2: counter clockwise 180.     i=3: counter clockwise 270.
        movements[:8] = np.roll(movements[:8], -2 * i)
        return movements



    def run(self):
        traj_gen = self.traj_generator()
        iteration = 0

        ### Stores previous iterations training data to augment current iteration data.
        ### May be useful, may be counterproductive. 
        previous_traj = {}
        max_avg = -10
        for i in range(100000):
        
            ### 0 if training selector, 1 if training mover
            turn = i % 2
            iteration = i + 1
            
            print("\n================= iteration {} =================".format(iteration))
            
            traj = traj_gen.__next__()
            
            self.add_vtarg_and_adv(traj)
            self.session.run(self.assign_op)
            
            
            #traj["advantage"] = (traj["advantage"] - np.mean(traj["advantage"])) / np.std(traj["advantage"])
            
            # Augment data
            #super_traj = self.rotateReflectAugmentation(traj)
            #traj = {}
            super_traj = traj
            traj = {}
            # normalize adv.
            
            
            len = int(super_traj["move_ob"].shape[0] / self.batch_size)
            
            if (turn == 1):
                print("\n================= Training selector =================")
            elif (turn == 0):
                print("\n================= Training mover =================")
            for _ in range(self.epochs):
                vf_loss = 0
                pol_loss = 0
                entropy = 0
                
                index_order = list(range(self.batch_size * len))
                np.random.shuffle(index_order)

                
                for i in range(len):
                    cur = i*self.batch_size
                    upper = cur+self.batch_size
                    curr_indices = index_order[cur:upper]    
                    #print(curr_indices)
                    #curr_indices = range(cur, upper)
                    
                    ### Handles mover training
                    if (turn == 0):
                        
                    
                        input_list = [self.move_ent, self.vf_loss, self.move_pol_loss, self.move_update_op]
                        
                        self.state_normalize(super_traj["move_ob"][curr_indices]) 
                        
                        input_dict = {
                                    self.obs_place: super_traj["move_ob"][curr_indices],
                                    self.acts_place: super_traj["move_action"][curr_indices],
                                    self.adv_place: super_traj["advantage"][curr_indices],
                                    self.return_place: super_traj["return"][curr_indices]
                                     }
                        
                                     
                        *step_losses, _ = self.session.run(input_list,
                                                        feed_dict=input_dict)
                        
                        ### Debug print statement for exploding value function
                        #print(self.session.run(self.net.v, feed_dict={self.obs_place: traj["move_ob"][cur:upper]}), traj["return"][cur:upper])
                           
                    ### Handles selector training
                    elif (turn == 1):
                        
                        input_list = [self.select_ent, self.vf_loss, self.select_pol_loss, self.select_update_op]
                        
                        self.state_normalize(super_traj["select_ob"][curr_indices])
                        
                        
                        input_dict = {
                                  self.obs_place: super_traj["select_ob"][curr_indices],
                                  self.select_acts_place: super_traj["select_action"][curr_indices],
                                  self.adv_place: super_traj["advantage"][curr_indices],
                                  self.return_place: super_traj["return"][curr_indices],
                                  self.tl_place: super_traj["tl_plc_in"][curr_indices]
                                     }
                        
                        
                        *step_losses, _ = self.session.run(input_list,
                                                        feed_dict=input_dict)
                        
                        ### Debug print statement for exploding value function
                        #print(self.session.run(self.net.v, feed_dict={self.obs_place: traj["select_ob"][cur:upper]}), traj["return"][cur:upper])
                     
                    entropy += step_losses[0] / len
                    vf_loss += step_losses[1] / len
                    pol_loss += step_losses[2] / len
                    
                ### Print training statistics for selector training    
                if (turn == 1):
                    print("vf_loss: {:.5f}, select_pol_loss: {:.5f}, entropy: {:.5f}".format(vf_loss, pol_loss, entropy))
                
                ### Print training statistics for mover training
                elif (turn == 0):
                    print("vf_loss: {:.5f}, pol_loss: {:.5f}, entropy: {:.5f}".format(vf_loss, pol_loss, entropy))
                    
            super_traj = {}
            traj = {}  
            n = 1
            # Save model every n iterations
            if iteration % n == 0 and self.averages[-1] > max_avg:
                max_avg = self.averages[-1]
                print(" Model saved ")
                self.save_model("./model_" + self.env.map + "/ppo_" + self.env.map)
            self.plot_results()
    
            
    def select_update(self):
        
        ent = self.select_entropy(self.net, self.batch_size)
        #ratio = tf.exp(self.select_logp(self.net) - tf.stop_gradient(self.select_logp(self.old_net)))
        #ratio = self.net.select_p / (self.old_net.select_p + self.epsilon)
        
        pol_surr = 0
        for i in range(len(self.net.select_p)):
            ratio = tf.exp(tf.log(tf.boolean_mask(self.net.select_p[i], self.select_acts_place[:,i,:])) - tf.log(tf.boolean_mask(self.old_net.select_p[i], self.select_acts_place[:,i,:]) + self.epsilon))
            surr1 = ratio * self.adv_place
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place
            if i == 0:
                pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
            else:
                pol_surr += (-tf.reduce_mean(tf.minimum(surr1, surr2))) 
        
        pol_surr /= 4
        #vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place))
        vf_loss = tf.losses.huber_loss(self.net.v, tf.reshape(self.return_place, [-1,1]))
        
        total_loss = pol_surr + self.c1 *vf_loss - self.c2 * ent
        
        update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)
        
        return ent, pol_surr, vf_loss, update_op

    def move_update(self):
        
        ent = self.move_entropy(self.net, self.batch_size)
        #ratio = tf.exp(self.move_logp(self.net) - tf.stop_gradient(self.move_logp(self.old_net)))
        ratio = tf.exp(   tf.log(tf.boolean_mask(self.net.p, self.acts_place)) - tf.log(tf.boolean_mask(self.old_net.p, self.acts_place) + self.epsilon)   )
        #print(ratio.shape)
        surr1 = ratio * self.adv_place
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place
        #print(self.net.p.shape, self.acts_place.shape)
        #print(ratio.shape, surr1.shape, surr2.shape)

        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2)) # -average(SUM RATIOn * ADVn)
        #vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place)) # -KL
        vf_loss = tf.losses.huber_loss(self.net.v, tf.reshape(self.return_place, [-1,1]))    
        total_loss = pol_surr + self.c1 *vf_loss - self.c2 * ent

        # Maximizing objective is same as minimizing the negative objective
        update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        return ent, pol_surr, vf_loss, update_op

    def add_vtarg_and_adv(self, traj):
        done = np.append(traj["done"], 0)
        value = np.append(traj["value"], traj["nextvalue"])
        T = len(traj["reward"])
        traj["advantage"] = gaelam = np.empty(T, 'float32')
        reward = traj["reward"]
        lastgaelam = 0

        for t in reversed(range(T)):
            nonterminal = 1 - done[t+1]
            delta = reward[t] + self.gamma * value[t+1] * nonterminal - value[t]
            gaelam[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        traj["return"] = traj["advantage"] + traj["value"]

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)
        print("model saved")

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
        print("model restored")

    def plot_results(self):
        plt.figure(1)
        plt.clf()
        plt.suptitle('Select-Move PPO')
        plt.title('Agent trained by Ray Sun, David Long, Michael McGuire', fontsize=7)
        plt.xlabel('Training iteration - DefeatRoaches')
        plt.ylabel('Average score')
        plt.plot(self.averages)
        plt.pause(0.001)  # pause a bit so that plots are updated
        
    def state_reshape(self, ob):
        return np.swapaxes(np.swapaxes(ob, 0, 1), 1, 2)
        
    def displayStack0(self, state):
        for i in range(state.shape[0]):        
            self.displayImage(state[i,:,:]) 
            
    def displayStack2(self, state):
        for i in range(state.shape[2]):
            self.displayImage(state[:,:,i])
        

    def displayImage(self, image):
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    env = MinigameEnvironment(state_modifier.modified_state_space,
                                map_name_="DefeatRoaches", 
                                render=False, 
                                step_multiplier=8)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    ppo = PPOAgent(env, session=sess)
    sess.run(tf.global_variables_initializer())
    #ppo.restore_model("./model_" + env.map + "/ppo_" + env.map)
    ppo.run()

    env.close()
