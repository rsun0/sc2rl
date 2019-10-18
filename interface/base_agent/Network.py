import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmarks = True
import numpy as np
import time

from agent import Model

from base_agent.sc2env_utils import generate_embeddings, multi_embed, valid_args, get_action_args, is_spatial_arg, env_config, processed_feature_dim
from base_agent.net_utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler, Unsqueeze, Squeeze, FastEmbedding

class BaseNetwork(nn.Module, Model):

    """
        IMPLEMENT THIS FUNCTION IN YOUR OWN NETWORK

        net_config: optional dictionary you can use to customize your network parameters
        device: Specify if you want the model on GPU or CPU

        Set it up in accordance with forward.
    """
    def __init__(self, net_config, device="cpu"):
        super(BaseNetwork, self).__init__()

    """
        IMPLEMENT THIS FUNCTION IN YOUR OWN NETWORK

        INPUTS:

        minimap: (N, D, H, W)
        screen: (N, D, H, W)
        player: (N, D)
        last_action: (N, D)
        hidden: (N, 2, D, H, W)
        curr_action: (N, D) (curr_action is not None if and only if training)
        choosing: True if sampling an action, False otherwise

        D is a placeholder for feature dimensions.

        OUTPUTS:

        action_logits: base action probabilities, shape is (batch_size, env_config["action_space"])
        arg_logits: arg probabilities, shape is (batch_size, env_config["max_arg_size"])
        spatial_logits: spatial arg probabilities, shape is (batch_size,
                                                            env_config["spatial_action_depth",
                                                            env_config["spatial_action_size"],
                                                            env_config["spatial_action_size"])
        hidden: hidden state of lstm, shape is (batch_size,
                                                2,
                                                net_config["LSTM_hidden_size"],
                                                LSTM_height,
                                                LSTM_width)
        value: value estimation of current state, shape is (batch_size, 1)
        choice: if batch_size = 1 and choosing==True, is list of format [int,
                                                        numpy array of length 10,
                                                        numpy array of shape (3,2)]
                else choice is None
    """
    def forward(self, minimap, screen, player, avail_actions, last_action, hidden, curr_action=None, choosing=False, unrolling=False):
        #return action_logits, arg_logits, spatial_logits, hidden, value, choice
        raise NotImplementedError


    def stacked_past_forward(self, minimaps, screens, players, avail_actions, last_actions, last_spatials, hiddens, curr_actions, relevant_frames):
        minimaps, screens, inputs2d = self.process_states(minimaps, screens, players[:,-1], last_actions[:,-1], last_spatials, sequential=False)
        #minimaps = minimaps.flatten(1, -1)
        #screens = screens.flatten(1, -1)
        action_logits, arg_logits, spatial_logits, _, values, _ = self.forward(
            minimaps,
            screens,
            players[:,-1],
            avail_actions,
            last_actions[:,-1],
            last_spatials,
            None,
            curr_action=curr_actions,
            process_inputs=False,
            format_inputs=False,
            inputs2d=inputs2d
        )

        return action_logits, arg_logits, spatial_logits, _, values, _

    def unroll_forward(self, minimaps, screens, players, avail_actions, last_actions, last_spatials, hiddens, curr_actions, relevant_frames):
        t1 = time.time()
        processed_minimaps, processed_screens, inputs2d = self.process_states(minimaps, screens, players, last_actions, last_spatials, sequential=False)
        hist_size = minimaps.shape[1]
        for i in range(hist_size-1):
            t5 = time.time()
            hiddens = self.forward(processed_minimaps[:,i],
                                        processed_screens[:,i],
                                        players[:,i],
                                        None,
                                        last_actions[:,i],
                                        last_spatials[:,i],
                                        hiddens,
                                        curr_action=None,
                                        unrolling=True,
                                        process_inputs=False,
                                        inputs2d=inputs2d[:,i])
            t6 = time.time()
            irrelevant_mask = relevant_frames[:,i] == 0
            hiddens[irrelevant_mask] = self.init_hidden(batch_size=torch.sum(irrelevant_mask), device=self.device)
            t7 = time.time()

        t2 = time.time()
        action_logits, arg_logits, spatial_logits, _, values, _ = self.forward(
                                                                                processed_minimaps[:,-1],
                                                                                processed_screens[:,-1],
                                                                                players[:,-1],
                                                                                avail_actions,
                                                                                last_actions[:,-1],
                                                                                last_spatials[:,-1],
                                                                                hiddens,
                                                                                curr_action=curr_actions,
                                                                                process_inputs=False,
                                                                                inputs2d=inputs2d[:,-1]
                                                                            )
        t3 = time.time()
        #print("Unroll time: %f. Regular forward time: %f. Total: %f" % (t2-t1, t3-t2, t3-t1))
        return action_logits, arg_logits, spatial_logits, _, values, _


    def unroll_forward_sequential(self, minimaps, screens, players, avail_actions, last_actions, last_spatial_actions, hiddens, curr_actions, relevant_frames, batch_size=32):
        t1 = time.time()
        N = minimaps.shape[0]
        hist_size = N - batch_size
        processed_minimaps, processed_screens, inputs2d = self.process_states(minimaps, screens, players, last_actions, last_spatial_actions)
        hiddens = hiddens[:batch_size]
        t2 = time.time()
        for i in range(hist_size-1):
            t5 = time.time()
            hiddens = self.forward(processed_minimaps[i:i+batch_size],
                                        processed_screens[i:i+batch_size],
                                        players[i:i+batch_size],
                                        None,
                                        last_actions[i:i+batch_size],
                                        last_spatial_actions[i:i+batch_size],
                                        hiddens,
                                        curr_action=None,
                                        unrolling=True,
                                        process_inputs=False,
                                        inputs2d=inputs2d[i:i+batch_size]
                                        )
            t6 = time.time()

            irrelevant_mask = relevant_frames[i:i+batch_size] == 0
            t7 = time.time()
            #print(irrelevant_mask)
            hiddens[irrelevant_mask] = self.init_hidden(batch_size=torch.sum(irrelevant_mask), device=self.device)
            t8 = time.time()

            #print("forward: %f. irrelevant: %f. hiddens: %f" % (t6-t5, t7-t6, t8-t7))

        t3 = time.time()
        action_logits, arg_logits, spatial_logits, _, values, _ = self.forward(
                                                                                processed_minimaps[-batch_size:],
                                                                                processed_screens[-batch_size:],
                                                                                players[-batch_size:],
                                                                                avail_actions[-batch_size:],
                                                                                last_actions[-batch_size:],
                                                                                last_spatial_actions[-batch_size:],
                                                                                hiddens,
                                                                                curr_action=curr_actions[-batch_size:],
                                                                                process_inputs=False,
                                                                                inputs2d=inputs2d[-batch_size:]
                                                                            )
        t4 = time.time()
        #print("Preprocess time: %f. Unroll time: %f. Regular forward time: %f. Total: %f" % (t2-t1, t3-t2, t4-t3, t4-t1))
        return action_logits, arg_logits, spatial_logits, _, values, _

    def process_states(self, minimaps, screens, players, last_actions, last_spatial_actions, sequential=True, embeddings_only=False):


        if not sequential:
            minimaps_s = minimaps.shape
            screens_s = screens.shape
            players_s = players.shape
            last_actions_s = last_actions.shape
            last_spatial_actions_s = last_spatial_actions.shape

            minimaps = minimaps.flatten(0,1)
            screens = screens.flatten(0,1)
            #players = players.flatten(0,1)
            #last_actions = last_actions.flatten(0,1)
            local_last_spatial_actions = last_spatial_actions.flatten(0,1)

        inputs = [minimaps, screens]
        [minimap, screen] = self.embed_inputs(inputs)
        [minimap, screen] = [minimap.to(self.device), screen.to(self.device)]
        [minimap, screen] = self.concat_spatial([minimap, screen], last_actions, local_last_spatial_actions)
        processed_minimap = minimap.view(minimaps_s[:2] + minimap.shape[1:])
        processed_screen = screen.view(screens_s[:2] + screen.shape[1:])
        processed_minimap = processed_minimap.flatten(1,2)
        processed_screen = processed_screen.flatten(1,2)

        if embeddings_only:
            return processed_minimap, processed_screen, None




        processed_minimap = self.down_layers_minimap(processed_minimap)
        processed_screen = self.down_layers_screen(processed_screen)

        embedded_last_actions = self.action_embedding(last_actions).to(self.device)
        nonspatial_concat = torch.cat([players, embedded_last_actions], dim=-1)
        inputs2d = self.inputs2d_MLP(nonspatial_concat)
        """
        if not sequential:
            processed_minimap = processed_minimap.view(minimaps_s[:2] + processed_minimap.shape[1:])
            processed_screen = processed_screen.view(screens_s[:2] + processed_screen.shape[1:])
            inputs2d = inputs2d.view(players_s[:2] + inputs2d.shape[1:])

            processed_minimap = processed_minimap.flatten(1,2)
            processed_screen = processed_screen.flatten(1,2)
            inputs2d = inputs2d.flatten(1,-1)
        """
        return processed_minimap, processed_screen, inputs2d

    def init_hidden(self, batch_size=1, use_torch=True, device="cuda:0"):
        return self.convLSTM.init_hidden_state(batch_size=batch_size, use_torch=use_torch, device=device)

    def embed_inputs(self, inputs):
        return multi_embed(inputs, self.state_embeddings, self.state_embedding_indices)


    def sample_func(self, probs):
        probs = probs.detach().cpu().numpy().astype(np.float64)
        assert (np.sum(probs) > 0)
        probs = probs / np.sum(probs)
        return np.argmax(np.random.multinomial(1, probs))

    def sample_action(self, action_logits):
        action = self.sample_func(action_logits[0])
        return action

    """
        arg_logit_inputs: (N, num_args) shape
    """
    def sample_arg(self, arg_logits, action):
        arg_out = np.zeros(self.arg_depth, dtype=np.int64)
        arg_types = self.action_to_nonspatial_args(action)
        for i in arg_types:
            arg_out[i-3] = self.sample_func(arg_logits[0,i-3])
        return arg_out

    def sample_spatial(self, spatial_logits, action):
        spatial_arg_out = - np.ones((self.spatial_depth, 2), dtype=np.int64)
        arg_types = self.action_to_spatial_args(action)

        for i in arg_types:
            spatial_probs_flat = spatial_logits[0,i].flatten()
            arg_index = self.sample_func(spatial_probs_flat)
            spatial_arg_out[i] = np.array([int(arg_index / self.spatial_size),
                                            arg_index % self.spatial_size])
        return spatial_arg_out

    def generate_arg_logits(self, arg_logit_inputs):
        initial_logits = F.softmax(arg_logit_inputs) * self.valid_args
        final_logits = initial_logits / torch.sum(initial_logits, dim=-1).unsqueeze(2)
        return final_logits

    def generate_spatial_logits(self, spatial_logits_in):
        (N, D, H, W) = spatial_logits_in.shape
        x = spatial_logits_in.flatten(start_dim=-2, end_dim=-1)
        logits = F.softmax(x, dim=-1)
        logits = logits.reshape((N, D, H, W))
        return logits

    def action_to_nonspatial_args(self, action):
        args = get_action_args(action)
        args = [i for i in args if not is_spatial_arg(i)]
        return args

    def action_to_spatial_args(self, action):
        args = get_action_args(action)
        args = [i for i in args if is_spatial_arg(i)]
        return args



















#############################
