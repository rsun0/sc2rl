"""
    Created by Michael McGuire, 07/31/2019

    This file, originally adapted from the experiments/DeepMind2018 agent,
    outlines a general agent that is capable of seeing the full minimap, screen,
    and player variables while accessing the full action space. Later it will be
    able to customize exactly which actions are available to the agent.

    Unless you want to really change the agent action selection logic or
    training algorithm, you shouldn't change this
    file. Currently, it is set up to use a network that takes in screen,
    minimap, player, a convolutional LSTM hidden state, and action information
    as shown in train_step. The variables it returns are action probabilities
    for base actions, args, and spatial args respectively, followed by hidden
    state, value, and choice.

"""




from agent import Agent, Model, Memory, AgentSettings
from base_agent.sc2env_utils import batch_get_action_args, is_spatial_arg

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import numpy as np
import time
import math
import matplotlib.pyplot as plt

import copy


class BaseAgent(Agent):

    def __init__(self, model, settings, memory, train_settings):
        super().__init__(model, settings, memory)
        self.step = 0
        self.frame_count = 0
        self.epochs_trained = 0
        self.train_settings = train_settings
        self.target_model = copy.deepcopy(model)
        self.hidden_state = self.model.init_hidden(use_torch=False)
        self.prev_hidden_state = None
        self.action = [0, np.zeros(10), np.zeros((3,2))]
        self.device = train_settings["device"]
        self.loss = nn.MSELoss()
        self.map = train_settings["map"]

    def _forward(self, agent_state, choosing=True):
        (minimap, screen, player, avail_actions) = agent_state
        self.prev_hidden_state = copy.deepcopy(self.hidden_state)
        _, _, _, self.hidden_state, value, action = self.model(minimap,
                                                            screen,
                                                            player,
                                                            avail_actions,
                                                            np.array([self.action[0]]),
                                                            self.hidden_state,
                                                            choosing=choosing)
        return _, _, value.cpu().data.numpy().item(), self.hidden_state.cpu().data.numpy(), action

    def _sample(self, agent_state):
        _, _, self.value, self.hidden_state, self.action = self._forward(agent_state, choosing=True)
        self.frame_count += 1
        self.step += 1
        return self.action

    def state_space_converter(self, state):
        return state

    def action_space_converter(self, personal_action):
        return personal_action

    def train(self, run_settings):
        self.memory.compute_vtargets_adv(self.train_settings['discount_factor'],
                                            self.train_settings['lambda'])

        batch_size = run_settings.batch_size
        num_iters = int(len(self.memory) / batch_size)
        epochs = run_settings.num_epochs

        for i in range(epochs):

            pol_loss = 0
            vf_loss = 0
            ent_total = 0

            for j in range(num_iters):

                d_pol, d_vf, d_ent = self.train_step_sequential(batch_size)
                if (math.isnan(d_pol) or math.isnan(d_vf) or math.isnan(d_ent)):
                    print("Canceling training -- NaN's encountered")
                    print("Reloading model from previous save")
                    self.load()
                    self.update_target_net()
                    return
                pol_loss += d_pol
                vf_loss += d_vf
                ent_total += d_ent

            self.epochs_trained += 1
            pol_loss /= num_iters
            vf_loss /= num_iters
            ent_total /= num_iters
            print("Epoch %d: Policy loss: %f. Value loss: %f. Entropy %f" %
                            (self.epochs_trained,
                            pol_loss,
                            vf_loss,
                            ent_total)
                            )
        self.update_target_net()

        print("\n\n ------- Training sequence ended ------- \n\n")

    def train_step(self, batch_size):

        t1 = time.time()

        device = self.train_settings['device']
        eps_denom = self.train_settings['eps_denom']
        c1 = self.train_settings['c1']
        c2 = self.train_settings['c2']
        clip_param = self.train_settings['clip_param']

        mini_batch = self.memory.sample_mini_batch(self.frame_count)
        t2 = time.time()
        n = len(mini_batch)
        mini_batch = np.array(mini_batch).transpose()

        states = np.stack(mini_batch[0], axis=0)
        actions = np.stack(np.array(mini_batch[1]), axis=0)
        minimaps = np.stack(states[:,0], axis=0).squeeze(2)
        screens = np.stack(states[:,1], axis=0).squeeze(2)
        hiddens = np.concatenate(states[:,4], axis=0)
        spatial_args = np.stack(actions[:,2], 0).astype(np.int64)
        minimaps, screens, hiddens, spatial_args = self.memory.batch_random_transform(minimaps, screens, hiddens, spatial_args)

        minimaps = torch.from_numpy(minimaps.copy()).float().to(self.device)
        screens = torch.from_numpy(screens.copy()).float().to(self.device)
        players = torch.from_numpy(np.stack(states[:,2], axis=0).squeeze(2)).float().to(self.device)
        avail_actions = torch.from_numpy(np.stack(states[:,3], axis=0)).byte().to(self.device)
        hidden_states = torch.from_numpy(hiddens.copy()).float().to(self.device)
        old_hidden_states = torch.from_numpy(np.concatenate(states[:,5], axis=0)).float().to(self.device)
        prev_actions = torch.from_numpy(np.stack(states[:,6], axis=0)).long().to(self.device).squeeze(1)
        relevant_states = torch.from_numpy(np.stack(states[:,7], axis=0)).byte().to(self.device)


        base_actions = np.stack(actions[:,0], 0).astype(np.int64).squeeze(1)
        args = np.stack(actions[:,1], 0).astype(np.int64)

        rewards = np.array(list(mini_batch[2]))
        dones = mini_batch[3]
        v_returns = mini_batch[5].astype(np.float32)
        advantages = mini_batch[6].astype(np.float32)

        rewards = torch.from_numpy(rewards).float().to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std()
        v_returns = torch.from_numpy(v_returns).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).byte().to(self.device)
        t3 = time.time()

        # minimaps, screens, players, avail_actions, last_actions, hiddens, curr_actions, relevant_frames
        action_probs, arg_probs, spatial_probs, _, values, _ = self.model.unroll_forward(
            minimaps,
            screens,
            players,
            avail_actions,
            prev_actions,
            hidden_states,
            base_actions,
            relevant_states
        )

        old_action_probs, old_arg_probs, old_spatial_probs, _, _, _ = self.target_model.unroll_forward(
            minimaps[:,[-1]],
            screens[:,[-1]],
            players[:,[-1]],
            avail_actions,
            prev_actions[:,[-1]],
            old_hidden_states,
            base_actions,
            relevant_states[:,[-1]]
        )
        t4 = time.time()

        gathered_actions = action_probs[range(n), base_actions]
        old_gathered_actions = old_action_probs[range(n), base_actions]

        gathered_args, old_gathered_args = self.index_args(arg_probs, old_arg_probs, args)

        gathered_spatial_args, old_gathered_spatial_args = self.index_spatial(spatial_probs,
                                                                            old_spatial_probs,
                                                                            spatial_args)

        action_args = batch_get_action_args(base_actions)
        """
        numerator = torch.zeros((n,)).float().to(self.device)
        denominator = torch.zeros((n,)).float().to(self.device)
        entropy = torch.zeros((n,)).float().to(self.device)
        """

        numerator = torch.log(gathered_actions)
        denominator = torch.log(old_gathered_actions + eps_denom)
        entropy = self.entropy(gathered_actions)
        num_args = torch.ones(n,).to(self.device)



        for i in range(n):
            curr_args = action_args[i]
            for j in curr_args:
                if is_spatial_arg(j):
                    numerator[i] = numerator[i] + torch.log(gathered_spatial_args[i][j])
                    denominator[i] = denominator[i] + torch.log(old_gathered_spatial_args[i][j] + eps_denom)
                    entropy[i] = entropy[i] + torch.mean(self.entropy(gathered_spatial_args[i][j]))
                else:
                    numerator[i] = numerator[i] + torch.log(gathered_args[i][j-3])
                    denominator[i] = denominator[i] + torch.log(old_gathered_args[i][j-3] + eps_denom)
                    entropy[i] = entropy[i] + torch.mean(self.entropy(gathered_args[i][j-3]))
            num_args[i] += len(curr_args)

        denominator = denominator.detach()
        t5 = time.time()

        #print(numerator, denominator, num_args)

        ratio = torch.exp((numerator - denominator) * (1 / num_args))
        ratio_adv = ratio * advantages.detach()
        bounded_adv = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        bounded_adv = bounded_adv * advantages.detach()

        pol_avg = - ((torch.min(ratio_adv, bounded_adv)).mean())
        value_loss = self.loss(values.squeeze(1), v_returns.detach())
        ent = entropy.mean()
        t6 = time.time()

        total_loss = pol_avg + c1 * value_loss + c2 * ent
        #total_loss = value_loss
        #total_loss = ent
        self.optimizer.zero_grad()
        total_loss.backward()
        #self.process_gradients(self.model)
        clip_grad_norm_(self.model.parameters(), 30.0)
        self.optimizer.step()
        t7 = time.time()
        pol_loss = pol_avg.detach().item()
        vf_loss = value_loss.detach().item()
        ent_total = ent.detach().item()
        #print("%f %f %f %f %f %f, total: %f" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t1))

        return pol_loss, vf_loss, -ent_total

    def train_step_sequential(self, batch_size):

        t1 = time.time()

        device = self.train_settings['device']
        eps_denom = self.train_settings['eps_denom']
        c1 = self.train_settings['c1']
        c2 = self.train_settings['c2']
        clip_param = self.train_settings['clip_param']

        states, mini_batch = self.memory.sample_mini_batch(self.frame_count, self.train_settings["hist_size"])
        t2 = time.time()
        n = len(mini_batch)
        mini_batch = np.array(mini_batch).transpose()

        actions = np.stack(np.array(mini_batch[0]), axis=0)
        minimaps = states[0]
        screens = states[1]
        players = states[2]
        hiddens = states[4]
        spatial_args = np.stack(actions[:,2], 0).astype(np.int64)
        minimaps, screens, hiddens, spatial_args = self.memory.batch_random_transform(minimaps, screens, hiddens, spatial_args)

        avail_actions = states[3].byte()
        old_hidden_states = hiddens[-batch_size:]
        prev_actions = states[5]
        relevant_states = states[6].byte()
        hidden_states = hiddens[:batch_size]

        base_actions = np.stack(actions[:,0], 0).astype(np.int64).squeeze(1)
        args = np.stack(actions[:,1], 0).astype(np.int64)

        rewards = np.array(list(mini_batch[1]))
        dones = mini_batch[2]
        v_returns = mini_batch[4].astype(np.float32)
        advantages = mini_batch[5].astype(np.float32)

        rewards = torch.from_numpy(rewards).float().to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std()
        v_returns = torch.from_numpy(v_returns).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).byte().to(self.device)
        t3 = time.time()
        base_actions = torch.from_numpy(base_actions).to(self.device)

        # minimaps, screens, players, avail_actions, last_actions, hiddens, curr_actions, relevant_frames
        action_probs, arg_probs, spatial_probs, _, values, _ = self.model.unroll_forward_sequential(
            minimaps,
            screens,
            players,
            avail_actions,
            prev_actions,
            hidden_states,
            base_actions,
            relevant_states
        )

        old_action_probs, old_arg_probs, old_spatial_probs, _, _, _ = self.target_model.unroll_forward_sequential(
            minimaps[-batch_size:],
            screens[-batch_size:],
            players[-batch_size:],
            avail_actions[-batch_size:],
            prev_actions[-batch_size:],
            old_hidden_states,
            base_actions[-batch_size:],
            relevant_states[-batch_size:]
        )
        t4 = time.time()

        gathered_actions = action_probs[range(batch_size), base_actions[-batch_size:]]
        old_gathered_actions = old_action_probs[range(batch_size), base_actions[-batch_size:]]

        args = args[-batch_size:]
        spatial_args = spatial_args[-batch_size:]

        gathered_args, old_gathered_args = self.index_args(arg_probs, old_arg_probs, args)

        gathered_spatial_args, old_gathered_spatial_args = self.index_spatial(spatial_probs,
                                                                            old_spatial_probs,
                                                                            spatial_args)

        action_args = batch_get_action_args(base_actions[-batch_size:])
        """
        numerator = torch.zeros((n,)).float().to(self.device)
        denominator = torch.zeros((n,)).float().to(self.device)
        entropy = torch.zeros((n,)).float().to(self.device)
        """

        numerator = torch.log(gathered_actions)
        denominator = torch.log(old_gathered_actions + eps_denom)
        entropy = self.entropy(gathered_actions)
        num_args = torch.ones(batch_size,).to(self.device)



        for i in range(batch_size):
            curr_args = action_args[i]
            for j in curr_args:
                if is_spatial_arg(j):
                    numerator[i] = numerator[i] + torch.log(gathered_spatial_args[i][j])
                    denominator[i] = denominator[i] + torch.log(old_gathered_spatial_args[i][j] + eps_denom)
                    entropy[i] = entropy[i] + torch.mean(self.entropy(gathered_spatial_args[i][j]))
                else:
                    numerator[i] = numerator[i] + torch.log(gathered_args[i][j-3])
                    denominator[i] = denominator[i] + torch.log(old_gathered_args[i][j-3] + eps_denom)
                    entropy[i] = entropy[i] + torch.mean(self.entropy(gathered_args[i][j-3]))
            num_args[i] += len(curr_args)

        denominator = denominator.detach()
        t5 = time.time()

        #print(numerator, denominator, num_args)

        ratio = torch.exp((numerator - denominator) * (1 / num_args))
        ratio_adv = ratio * advantages.detach()[-batch_size:]
        bounded_adv = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        bounded_adv = bounded_adv * advantages.detach()[-batch_size:]

        pol_avg = - ((torch.min(ratio_adv, bounded_adv)).mean())
        value_loss = self.loss(values.squeeze(1), v_returns[-batch_size:].detach())
        ent = entropy.mean()
        t6 = time.time()

        total_loss = pol_avg + c1 * value_loss + c2 * ent
        #total_loss = value_loss
        #total_loss = ent
        self.optimizer.zero_grad()
        total_loss.backward()
        #self.process_gradients(self.model)
        clip_grad_norm_(self.model.parameters(), 100.0)
        self.optimizer.step()
        t7 = time.time()
        pol_loss = pol_avg.detach().item()
        vf_loss = value_loss.detach().item()
        ent_total = ent.detach().item()
        #print("%f %f %f %f %f %f, total: %f" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t1))

        return pol_loss, vf_loss, -ent_total





    def load(self):
        self.model.load_state_dict(torch.load("save_model/Starcraft2" + self.map + "RRL.pth"))
        self.update_target_net()

    def save(self):
        torch.save(self.model.state_dict(), "save_model/Starcraft2" + self.map + "RRL.pth")

    def push_memory(self, state, action, reward, done):
        push_state = list(state) + [self.prev_hidden_state]
        self.memory.push(push_state, action, reward, done, self.value, 0, 0, self.step)
        if done:
            self.step = 0
            self.value = 0
            self.hidden_state = self.model.init_hidden(use_torch=False)

    ### Unique RRL functions below this line

    """
        arg_probs: (N, 10, 500)
        old_arg_probs: ""
        args: (N, 10)

        returns: (N, 10), (N, 10)
    """
    def index_args(self, arg_probs, old_arg_probs, args):
        (N, D) = args.shape
        flattened = arg_probs.view(-1, arg_probs.shape[-1])
        old_flattened = old_arg_probs.view(-1, old_arg_probs.shape[-1])
        gathered = flattened[range(len(flattened)), args.flatten()]
        old_gathered = old_flattened[range(len(old_flattened)), args.flatten()]
        return gathered.reshape((N, D)), old_gathered.reshape((N, D))

    """
        spatial_probs: (N, 3, 64, 64)
        old_spatial_probs: ""
        spatial_args: (N, 3, 2)

        returns: (N, 3), (N, 3)
    """
    def index_spatial(self, spatial_probs, old_spatial_probs, spatial_args):
        (N, D, H, W) = spatial_probs.shape
        flattened = spatial_probs.view(-1, H, W)
        old_flattened = old_spatial_probs.view(-1, H, W)
        gathered = flattened[range(N*D), spatial_args[:,:,0].flatten(), spatial_args[:,:,1].flatten()]
        old_gathered = old_flattened[range(N*D), spatial_args[:,:,0].flatten(), spatial_args[:,:,1].flatten()]
        gathered = gathered.reshape((N, D))
        old_gathered = old_gathered.reshape((N, D))
        return gathered, old_gathered


    def update_target_net(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def entropy(self, x):
        output = torch.log(x + self.train_settings["eps_denom"]) * x
        return output


    ### Network processing functions

    def process_gradients(self, network, clip=1.0):
        grad_sum = 0
        for param in network.parameters():
            if param.grad is None:
                continue
            grad_sum += torch.sum(param.grad.data ** 2)
        print(grad_sum)
