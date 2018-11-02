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
    def __init__(self, env, scope, num_layers, num_units, obs_plc, act_plc, trainable=True):
        self.env = env
        self.observation_size = obs_plc # TODO:
        self.action_size = env.action_space
        self.trainable = trainable
        self.activation = tf.nn.relu
        self.scope = scope

        self.obs_place = obs_plc
        self.acts_place = act_plc

        self.p , self.v, self.logstd = self._build_network(num_layers=num_layers, num_units=num_units)
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
                x = tf.layers.dense(x, units=num_units, activation=self.activation, name="p_fc"+str(i),
                                    trainable=self.trainable)
            action = tf.layers.dense(x, units=self.action_size, activation=tf.nn.softmax,
                                     name="p_fc"+str(num_layers), trainable=self.trainable)


            x = self.obs_place
            
            x = tf.layers.conv2d(x,
                filters=32,
                kernel_size=[8,8],
                padding="same",
                strides=(4, 4),
                activation=self.activation)
                
            x = tf.layers.conv2d(x,
                filters=64,
                kernel_size=[4,4],
                padding="same",
                strides=(2, 2),
                activation=self.activation)
                
            x = tf.contrib.layers.flatten(x)
            
            
            
            for i in range(num_layers):
                x = tf.layers.dense(x, units=num_units, activation=self.activation, name="v_fc"+str(i),
                                    trainable=self.trainable)
            value = tf.layers.dense(x, units=1, activation=None, name="v_fc"+str(num_layers),
                                    trainable=self.trainable)

            logstd = tf.get_variable(name="logstd", shape=[self.action_size],
                                     initializer=tf.zeros_initializer)

        return action, value, logstd

    def action_sample(self):
        return self.p # + tf.exp(self.logstd) * tf.random_normal(tf.shape(self.p))

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
        self.learning_rate = 1e-4
        self.epochs = 3
        self.step_size = 2000
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

        ## build network
        self.net = Network(env=self.env,
                           scope="pi",
                           num_layers=2,
                           num_units=128,
                           obs_plc=self.obs_place,
                           act_plc=self.acts_place)

        self.old_net = Network(env=self.env,
                               scope="old_pi",
                               num_layers=2,
                               num_units=128,
                               obs_plc=self.obs_place,
                               act_plc=self.acts_place,
                               trainable=False)

        # tensorflow operators
        self.assign_op = self.assign(self.net, self.old_net)
        self.ent, self.pol_loss, self.vf_loss, self.update_op = self.update()
        self.saver = tf.train.Saver()

    @staticmethod
    def logp(net):
        print(net.acts_place, net.p, net.logstd)
        logp = -(0.5 * tf.reduce_sum(tf.square((net.acts_place - net.p) / tf.exp(net.logstd)), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(net.p)[-1]) \
            + tf.reduce_sum(net.logstd, axis=-1))

        return logp

    @staticmethod
    def entropy(net):
        ent = tf.reduce_sum(net.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return ent

    @staticmethod
    def assign(net, old_net):
        assign_op = []
        for (newv, oldv) in zip(net.get_variables(), old_net.get_variables()):
            assign_op.append(tf.assign(oldv, newv))

        return assign_op

    def traj_generator(self):
        t = 0
        action = int(env.action_space * random.random()) # this is replacement of env.action_space.sample()
        done = True
        ob, reward, done, _ = env.reset()
        ob = ob.reshape(self.input_shape)
        ob = self.normalize(ob)
        cur_ep_return = 0
        cur_ep_length = 0
        ep_returns = []
        ep_lengths = []

        scores = []
        num_games = 0

        obs = np.array([ob for _ in range(self.step_size)])
        rewards = np.zeros(self.step_size, 'float32')
        values = np.zeros(self.step_size, 'float32')
        dones = np.zeros(self.step_size, 'int32')
        actions = np.array([np.zeros((self.env.action_space,)) for _ in range(self.step_size)])
        prevactions = actions.copy()



        while True:
            prevaction = action
            action_vals, value = self.act(ob)
            action = self.select_action(action_vals)
            if t > 0 and t % self.step_size == 0:
                self.averages.append(sum(scores) / num_games)
                scores = []
                num_games = 0
                yield {"ob": obs, "reward":rewards, "value": values,
                       "done": dones, "action": actions, "prevaction": prevactions,
                       "nextvalue": value*(1 - done), "ep_returns": ep_returns,
                       "ep_lengths": ep_lengths}

                ep_returns = []
                ep_lengths = []

            i = t % self.step_size
            obs[i] = ob
            values[i] = value
            dones[i] = done
            actions[i] = np.zeros((self.env.action_space,))
            actions[i][action] = 1
            prevactions[i] = np.zeros((self.env.action_space,))
            prevactions[i][prevaction] = 1

            ob, reward, done, _ = env.step(action) # TODO: select argmax from action? or is action[0] always?
            ob = ob.reshape(self.input_shape)
            ob = self.normalize(ob)
            rewards[i] = reward

            cur_ep_return += reward
            cur_ep_length += 1

            if done:
                scores.append(cur_ep_return)
                num_games += 1
                print("Reward: {}".format(cur_ep_return))
                ep_returns.append(cur_ep_return)
                ep_lengths.append(cur_ep_length)
                cur_ep_return = 0
                cur_ep_length = 0
                ob, reward, done, _ = env.reset()
                ob = ob.reshape(self.input_shape)
                ob = self.normalize(ob)
            t += 1

    def act(self, ob):
        actions, value = self.session.run([self.net.act_op, self.net.v], feed_dict={
            self.net.obs_place: ob[None]
        })
        return actions, value
        
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
        
        

    def run(self):
        traj_gen = self.traj_generator()
        iteration = 0

        for _ in range(100000):
            iteration += 1
            print("\n================= iteration {} =================".format(iteration))
            traj = traj_gen.__next__()
            self.add_vtarg_and_adv(traj)

            self.session.run(self.assign_op)

            # normalize adv.
            traj["advantage"] = (traj["advantage"]-np.mean(traj["advantage"]))/np.std(traj["advantage"])

            len = int(self.step_size / self.batch_size)
            for _ in range(self.epochs):
                vf_loss = 0
                pol_loss = 0
                entropy = 0
                for i in range(len):
                    cur = i*self.batch_size
                    *step_losses, _ = self.session.run([self.ent, self.vf_loss, self.pol_loss, self.update_op],feed_dict = {self.obs_place: traj["ob"][cur:cur+self.batch_size],
                                                       self.acts_place: traj["action"][cur:cur+self.batch_size],
                                                       self.adv_place: traj["advantage"][cur:cur+self.batch_size],
                                                       self.return_place: traj["return"][cur:cur+self.batch_size]})

                    entropy += step_losses[0]/len
                    vf_loss += step_losses[1]/len
                    pol_loss += step_losses[2]/len
                print("vf_loss: {:.5f}, pol_loss: {:.5f}, entorpy: {:.5f}".format(vf_loss, pol_loss, entropy))

            # Save model every 100 iterations
            if iteration % 10 == 0:
                print(" Model saved ")
                self.save_model("./model_" + self.env.map + "/ppo_" + self.env.map)
            self.plot_results()

    def update(self):
        print("--- update called ---")
        
        ent = self.entropy(self.net)
        ratio = tf.exp(self.logp(self.net) - tf.stop_gradient(self.logp(self.old_net)))
        surr1 = ratio * self.adv_place
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place

        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2)) # -average(SUM RATIOn * ADVn)
        vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place)) # -KL

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
                                render=True, 
                                step_multiplier=2)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    ppo = PPOAgent(env, session=sess)
    sess.run(tf.global_variables_initializer())
    #ppo.restore_model("./model_" + env.map + "/ppo_" + env.map)
    ppo.run()

    env.close()
