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

class Network(object):
    def __init__(self, env, scope, num_layers, num_units, obs_plc, act_plc, select_act_plc, trainable=True):
        self.env = env
        self.observation_size = obs_plc # TODO:
        self.action_size = env.action_space
        self.select_size = env.select_space
        self.select_width = env.select_space
        self.select_height = env.select_space
        self.trainable = trainable
        self.activation = tf.nn.relu
        self.scope = scope

        self.obs_place = obs_plc
        self.acts_place = act_plc
        self.select_acts_place = select_act_plc

        self.p , self.v, self.select_p, self.logstd, self.select_logstd = self._build_network(num_layers=num_layers, num_units=num_units)
        self.act_op = self.action_sample()
        
        

    def _build_network(self, num_layers, num_units):
        # TODO: switch to CONV NETS
        with tf.variable_scope(self.scope):
            x = self.obs_place
            
            # Initializes convolutional layers
            
            x = tf.layers.conv2d(x,
                filters=32,
                kernel_size=[8, 8],
                padding="same",
                strides=(4, 4),
                activation=self.activation)
                
            x = tf.layers.conv2d(x,
                filters=64,
                kernel_size=[4, 4],
                padding="same",
                strides=2,
                activation=self.activation)
                
            x = tf.contrib.layers.flatten(x)
            
            # Initializes fully connected layers
            for i in range(num_layers):
                x = tf.layers.dense(x, 
                                units=num_units, 
                                activation=self.activation, 
                                name="p_fc"+str(i), 
                                trainable=self.trainable)
                                
            action = tf.layers.dense(x, 
                                units=self.action_size, 
                                activation=tf.nn.softmax,
                                name="p_fc"+str(num_layers), 
                                trainable=self.trainable)
                
                
                
            ### select_p network #####################################
            
            x = self.obs_place
            
            # Initializes convolutional layers
            
            x = tf.layers.conv2d(x,
                filters=32,
                kernel_size=[8, 8],
                padding="same",
                strides=(4, 4),
                activation=self.activation)
                
            x = tf.layers.conv2d(x,
                filters=64,
                kernel_size=[4, 4],
                padding="same",
                strides=2,
                activation=self.activation)
                         
            select_p = tf.layers.conv2d(x,
                filters=64,
                kernel_size=[3,3],
                padding="same",
                strides=1,
                activation=self.activation)
                
            ### Maybe add more dense layers
            select_p = tf.layers.dense(select_p, units=num_units, activation=self.activation, name="select_p_fc1", trainable=self.trainable)
                
            select_p = tf.contrib.layers.flatten(select_p)
            
            select_p_x1 = tf.layers.dense(select_p, units=self.select_width, activation=tf.nn.softmax, name="select_p_x1_fc", trainable=self.trainable)
            select_p_y1 = tf.layers.dense(select_p, units=self.select_height, activation=tf.nn.softmax, name="select_p_y1_fc", trainable=self.trainable)
            
            x2_y2_in = tf.concat([select_p, select_p_x1, select_p_y1], axis=-1)
  
            ### Maybe add dense layers          
            select_p_x2 = tf.layers.dense(x2_y2_in, units=self.select_width, activation=tf.nn.softmax, name="select_p_x2_fc", trainable=self.trainable)
            select_p_y2 = tf.layers.dense(x2_y2_in, units=self.select_height, activation=tf.nn.softmax, name="select_p_y2_fc", trainable=self.trainable)
            
            
                
            
                                     
            x = self.obs_place
            
            x = tf.layers.conv2d(x,
                                filters=32,
                                kernel_size=[8,8],
                                padding="same",
                                strides=(4,4),
                                activation=self.activation)
                                
            x = tf.layers.conv2d(x,
                                filters=64,
                                kernel_size=[4,4],
                                padding="same",
                                strides=(2,2),
                                activation=self.activation)
                                
            x = tf.contrib.layers.flatten(x)
            
            for i in range(num_layers):
                x = tf.layers.dense(x, units=num_units, activation=self.activation, name="v_fc"+str(i), trainable=self.trainable)
                
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
        ## hyperparameters - TODO: TUNE
        self.learning_rate = 5e-5
        self.epochs = 4
        self.step_size = 3000
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_param = 0.2
        self.batch_size = 32
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

        ## build network
        self.net = Network(env=self.env,
                           scope="pi",
                           num_layers=2,
                           num_units=512,
                           obs_plc=self.obs_place,
                           act_plc=self.acts_place,
                           select_act_plc=self.select_acts_place)

        self.old_net = Network(env=self.env,
                               scope="old_pi",
                               num_layers=2,
                               num_units=512,
                               obs_plc=self.obs_place,
                               act_plc=self.acts_place,
                               select_act_plc=self.select_acts_place,
                               trainable=False)

        # tensorflow operators
        self.assign_op = self.assign(self.net, self.old_net)
        self.select_ent, self.select_pol_loss, self.vf_loss, self.select_update_op = self.select_update()
        self.move_ent, self.move_pol_loss, self.vf_loss, self.move_update_op = self.move_update()
        self.saver = tf.train.Saver()

    @staticmethod
    def select_logp(net):
        logp = 0
        for i in range(4):
            p = net.select_p[i]
            logp += (-(0.5 * tf.reduce_sum(tf.square((net.select_acts_place[:,i] - p) / tf.exp(net.select_logstd)), axis=-1) \
                + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(p)[-1]) \
                + tf.reduce_sum(net.select_logstd, axis=-1)) )
        return logp / 4
        
    @staticmethod
    def move_logp(net):
        logp = -(0.5 * tf.reduce_sum(tf.square((net.acts_place - net.p) / tf.exp(net.logstd)), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(net.p)[-1]) \
            + tf.reduce_sum(net.logstd, axis=-1))

        return logp

    @staticmethod
    def move_entropy(net):
        ent = tf.reduce_sum(net.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return ent

    @staticmethod
    def select_entropy(net):
        ent = tf.reduce_sum(net.select_logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return ent

    @staticmethod
    def assign(net, old_net):
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
        ob = ob.reshape(self.input_shape)
        ob = self.normalize(ob)
        cur_ep_return = 0
        cur_ep_length = 0
        
        scores = []
        num_games = 0
        
        #obs = np.array([ob, ob]for _ in range(self.step_size)]).T
        #select_obs = np.array([ob for _ in range(self.step_size)])
        #move_obs = np.array([ob for _ in range(self.step_size)])
        select_obs = np.zeros((self.step_size, ob.shape[0], ob.shape[1], ob.shape[2]), 'float32')
        move_obs = np.zeros((self.step_size, ob.shape[0], ob.shape[1], ob.shape[2]), 'float32')
        rewards = np.zeros(self.step_size, 'float32')
        values = np.zeros(self.step_size, 'float32')
        dones = np.zeros(self.step_size, 'int32')
        select_actions = np.zeros((self.step_size, 4, self.env.select_space), 'float32')
        move_actions = np.zeros((self.step_size, self.env.action_space), 'float32')
        
        prev_select_actions = select_actions.copy()
        prev_move_actions = move_actions.copy()
        
        selection = 0
        movement = 0
        prev_selection = [-1 for i in range(4)]
        prev_movement = -1
        
        ep_returns = []
        ep_lengths = []
        
        while True:
        
            ### Stores index of current step
            i = int((t % (2 * self.step_size)) / 2)
        
            ### Even if selecting, odd if moving
            j = t % 2
        
            ### Handles return ###
            if (t > 0 and (t % (2*self.step_size)) == 0):
                self.averages.append(sum(scores) / (num_games))
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
                selection_vals, value = self.select(ob)
                selection_nums = self.select_selection(selection_vals)
                
                select_obs[i] = ob
                values[i] += value
                dones[i] = done
                select_actions[i] = np.zeros((4, self.env.select_space,))
                select_actions[i, range(4), selection_nums] = 1
                prev_select_actions[i] = np.zeros((4, self.env.select_space,))
                prev_select_actions[i, range(4), prev_selection] = 1
                
                #csa = converted_select_action = self.select_convert(selection_nums)
                ob, temp_reward, done, _ = self.env.step(0, topleft=selection_nums[:2], botright=selection_nums[2:])
                
                
                ob = ob.reshape(self.input_shape)
                ob = self.normalize(ob)
                reward += temp_reward
                
                cur_ep_return += reward
                cur_ep_length += 1
                
                ### Timestep finished on select move;
                ### Add reward, value to previous timestep reward, value
                if done:
                    
                    ### Cancel in case this is first step
                    if (i == 0):
                        continue
                        
                    t -= 1
                    i -= 1
                    values[i] += value
                    rewards[i] += reward
                    scores.append(cur_ep_return)
                    num_games += 1
                    print("Reward: {}. {} of {} steps.".format(cur_ep_return, t%(2*self.step_size), (2*self.step_size)))
                    ep_returns.append(cur_ep_return)
                    ep_lengths.append(cur_ep_length)
                    cur_ep_return = 0
                    cur_ep_length = 0
                    ob, reward, done, _ = env.reset()
                    ob = ob.reshape(self.input_shape)
                    ob = self.normalize(ob)
                    t += 1
                    i += 1
                    continue
                    
                selection = selection_nums
                    
                    
            ### Handles movement ###
            
            else:
                prev_movement = movement
                action_vals, value = self.act(ob)
                movement = self.select_action(action_vals)
                move_obs[i] = ob
                values[i] += value
                dones[i] = done
                move_actions[i] = np.zeros((self.env.action_space,))
                move_actions[i][movement] = 1
                prev_move_actions[i] = np.zeros((self.env.action_space,))
                prev_move_actions[i][prev_movement] = 1
                
                ob, temp_reward, done, _ = self.env.step(movement)
                ob = ob.reshape(self.input_shape)
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
                    ob = ob.reshape(self.input_shape)
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
        x1, y1, x2, y2, value = self.session.run(self.net.select_p + [self.net.v], feed_dict={ self.net.obs_place: ob[None]
        })
        return np.array([x1, y1, x2, y2]), value
        
    def normalize(self, ob):
        #return ((ob.T - np.mean(ob, axis=(1,2))) / (1 + np.max(ob, axis=(1,2)) - np.min(ob, axis=(1,2)))).T
        return ob
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
        selection_probs = selection_probs.reshape((4, self.env.select_space))
        for i in range(selection_probs.shape[0]):
            num = random.random()
            running_sum = 0.0
            for j in range(len(selection_probs[i])):
                running_sum += selection_probs[i][j]
                if num < running_sum:
                    output.append(j)
                    break
        return output
        
    
    def run(self):
        traj_gen = self.traj_generator()
        iteration = 0

        ### Stores previous iterations training data to augment current iteration data.
        ### May be useful, may be counterproductive. 
        previous_traj = {}
        
        for i in range(100000):
        
            ### 0 if training selector, 1 if training mover
            turn = i % 2
            iteration = i + 1
            
            print("\n================= iteration {} =================".format(iteration))
            
            traj = traj_gen.__next__()
            self.add_vtarg_and_adv(traj)
            self.session.run(self.assign_op)
            
            # normalize adv.
            traj["advantage"] = (traj["advantage"] - np.mean(traj["advantage"])) / np.std(traj["advantage"])
            
            len = int(self.step_size / self.batch_size)
            
            if (turn == 1):
                print("\n================= Training selector =================")
            elif (turn == 0):
                print("\n================= Training mover =================")
            for _ in range(self.epochs):
                vf_loss = 0
                pol_loss = 0
                entropy = 0
                
                for i in range(len):
                    cur = i*self.batch_size
                    upper = cur+self.batch_size
                    
                    ### Handles mover training
                    if (turn == 0):
                    
                        input_list = [self.move_ent, self.vf_loss, self.move_pol_loss, self.move_update_op]
                        input_dict = {
                                    self.obs_place: traj["move_ob"][cur:upper],
                                    self.acts_place: traj["move_action"][cur:upper],
                                    self.adv_place: traj["advantage"][cur:upper],
                                    self.return_place: traj["return"][cur:upper]
                                     }
                                     
                        *step_losses, _ = self.session.run(input_list,
                                                        feed_dict=input_dict)
                           
                    ### Handles selector training
                    elif (turn == 1):
                        
                        input_list = [self.select_ent, self.vf_loss, self.select_pol_loss, self.select_update_op]
                        input_dict = {
                                  self.obs_place: traj["select_ob"][cur:upper],
                                  self.select_acts_place: traj["select_action"][cur:upper],
                                  self.adv_place: traj["advantage"][cur:upper],
                                  self.return_place: traj["return"][cur:upper]
                                     }
                        
                        *step_losses, _ = self.session.run(input_list,
                                                        feed_dict=input_dict)
                     
                    entropy += step_losses[0] / len
                    vf_loss += step_losses[1] / len
                    pol_loss += step_losses[2] / len
                    
                ### Print training statistics for selector training    
                if (turn == 0):
                    print("vf_loss: {:.5f}, select_pol_loss: {:.5f}, entropy: {:.5f}".format(vf_loss, pol_loss, entropy))
                
                ### Print training statistics for mover training
                elif (turn == 1):
                    print("vf_loss: {:.5f}, pol_loss: {:.5f}, entropy: {:.5f}".format(vf_loss, pol_loss, entropy))
                    
                    
            # Save model every 10 iterations
            if iteration % 10 == 0:
                print(" Model saved ")
                self.save_model("./model_" + self.env.map + "/ppo_" + self.env.map)
            self.plot_results()
    
            
    def select_update(self):
        
        ent = self.select_entropy(self.net)
        ratio = tf.exp(self.select_logp(self.net) - tf.stop_gradient(self.select_logp(self.old_net)))
        surr1 = ratio * self.adv_place
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place
        
        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
        #vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place))
        vf_loss = tf.losses.huber_loss(self.net.v, tf.reshape(self.return_place, [-1,1]))
        
        total_loss = pol_surr + 10*vf_loss
        
        update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)
        
        return ent, pol_surr, vf_loss, update_op

    def move_update(self):
        
        ent = self.move_entropy(self.net)
        ratio = tf.exp(self.move_logp(self.net) - tf.stop_gradient(self.move_logp(self.old_net)))
        surr1 = ratio * self.adv_place
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place

        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2)) # -average(SUM RATIOn * ADVn)
        #vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place)) # -KL
        vf_loss = tf.losses.huber_loss(self.net.v, tf.reshape(self.return_place, [-1,1]))    

        total_loss = pol_surr + 10*vf_loss

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
        plt.suptitle('11-Action Single-unit PPO')
        plt.title('Agent trained by Ray Sun, David Long, Michael McGuire', fontsize=7)
        plt.xlabel('Training iteration - DefeatRoaches')
        plt.ylabel('Average score')
        plt.plot(self.averages)
        plt.pause(0.001)  # pause a bit so that plots are updated


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
