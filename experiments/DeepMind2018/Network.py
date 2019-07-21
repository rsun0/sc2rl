import torch
import torch.nn as nn
import torch.nn.Functional as F

from agent import Model

from sc2env_utils import generate_embeddings, multi_embed, valid_args, get_action_args, is_spatial_arg
from net_utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler, Unsqueeze

class Model(nn.Module, Model):

    """
        net_config = {
            "minimap_features": int, number of features in minimap image
            "screen_features": int, number of features in screen image
            "player": int, number of features in player variable
            "last_action_features": int, number of features in last_action variable
            "down_conv_features": int, numbe features coming out of first convolution
            "up_features": int, number of features going into first upsample
            "up_conv_features": int, number of features coming out of first upsample
            "relational_spatial_depth": int, number of features going into output conv
            "action_space": int, number of base actions to pick from
            "arg_depth": int, total number of nonspatial argument types
            "max_arg_size": int, maximum number of arg categories to pick from
            "spatial_action_depth": int, max number of spatial arguments for action
            "spatial_action_size": int, width and height of spatial action space
            "resnet_features": int, number of features in each convolutional layer
            "resnet_depth": int, number of convolutional layers in resnet block
            "LSTM_in_size": int, number of features in the input to the LSTM
            "LSTM_hidden_size:" int, number of features in the hidden state of LSTM
            "inputs2d_size": int, number of features in inputs2d variable
            "relational_features": int, number of features in each relational block
            "relational_depth": int, number of relational blocks to put in sequence
            "screen_categorical_indices": int array, binary mask. i'th index is 1
                                            iff i'th layer of screen is categorical
            "minimap_categorical_indices": Like screen_categorical_indices.
            "player_categorical_indices": Like screen_categorical_indices.
            "screen_categorical_size": int array, number of screen categories
                                        to embed, corresponds with *_indices
            "minimap_categorical_size": int array, number of mmap categories
                                        to embed, corresponds with *_indices
            "player_categorical_size": int array, number of player categories
                                        to embed, corresponds with *_indices
            "embedding_size": int, number of features output by embeddings
        }
    """
    def __init__(self, net_config):
        self.net_config = net_config

        self.embeddings = generate_embeddings(net_config)
        self.embedding_indices = [
            net_config["minimap_categorical_indices"],
            net_config["screen_categorical_indices"],
            net_config["player_categorical_indices"]
        ]

        self.down_layers_minimap = self.Downsampler(net_config['minimap_features'], net_config)
        self.down_layers_screen = self.Downsampler(net_config['screen_features'], net_config)

        self.convLSTM = ConvLSTM(self.net_config['LSTM_in_size'],
                                    self.net_config['LSTM_hidden_size'])

        self.spatial_upsampler = SpatialUpsampler(net_config)

        self.inputs2d_MLP = nn.Sequential(
            nn.Linear(net_config['player_features'] + net_config['last_action_features'],
                        128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.attention_blocks = nn.Sequential(
            SelfAttentionBlock(net_config['inputs2d_size'],
                                net_config['relational_features'])
        )
        for i in range(net_config['relational_depth']-1):
            self.attention_blocks.add_module("Block"+str(i+2),
                                                SelfAttentionBlock(
                                                    net_config['relational_features'],
                                                    net_config['relational_features']
                                                )
                                            )

        self.relational_processor = nn.Sequential(
            nn.MaxPool2d(net_config['inputs3d_width']),
            Unsqueeze(),
            Unsqueeze(),
            nn.Linear(net_config['inputs3d_width'],
                        512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )


        self.value_MLP = nn.Sequential(

        )
        self.action_MLP = nn.Sequential(nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, net_config["action_space"])
                                        )

        self.arg_MLP = nn.Linear(net_config["shared_conditioned_features"],
                                    net_config["arg_depth"]*net_config["max_arg_size"])

        self.arg_depth = net_config["arg_depth"]
        self.arg_size = net_config["max_arg_size"]
        self.spatial_depth = net_config["spatial_action_depth"]
        self.spatial_size = net_config["spatial_action_size"]

        self.valid_args = valid_args
        self.action_to_args = action_to_args


    """
        minimap: (N, D, H, W)
        screen: (N, D, H, W)
        player: (N, D)
        last_action: (N, D)
        hidden: (N, 2, D, H, W)
        curr_action: (N, D) (curr_action is not None if and only if training)
        choosing: True if sampling an action, False otherwise

        D is a placeholder for feature dimensions.
    """
    def forward(self, minimap, screen, player, avail_actions, last_action, hidden, curr_action=None, choosing=False):
        if (choosing):
            assert (curr_action is None)
        if (not choosing):
            assert (curr_action is not None)

        N = len(minimap)

        inputs = [minimap, screen]
        [minimap, screen] = self.embed_inputs(inputs, net_config)

        processed_minimap = self.down_layers_minimap(minimap)
        processed_screen = self.down_layers_screen(screen)
        inputs3d = torch.cat([processed_minimap, processed_screen], dim=1)

        embedded_last_action = self.embed_action(last_action)
        nonspatial_concat = torch.cat([player, embedded_last_action], dim=-1)
        inputs2d = self.inputs2d_MLP(nonspatial_concat)

        expanded_inputs2d = inputs2d.unsqueeze(2).unsqueeze(3).expand(inputs2d.shape + (self.spatial_size, self.spatial_size))
        LSTM_in = torch.cat([inputs3d, expanded_inputs2d], axis=1)

        outputs2d, hidden = self.convLSTM(LSTM_in, hidden)
        relational_spatial = self.attention_blocks(outputs2d)
        relational_nonspatial = self.relational_processor(relational_spatial)

        shared_features = torch.cat([inputs2d, relational_nonspatial], dim=-1)
        value = self.value_MLP(shared_features)
        action_logits_in = self.action_MLP(shared_features)
        action_logits_in.masked_fill(1-avail_actions, float('-inf'))
        action_logits = F.softmax(action_logits_in)
        #action_logits = action_logits / torch.sum(action_logits, axis=-1)

        choice = None
        if (choosing):
            action = self.sample_action(action_logits)
        else:
            action = curr_action

        embedded_action = self.embed_action(action)
        shared_conditioned = torch.cat([shared_features, embedded_action], dim=-1)
        arg_logit_inputs = self.arg_MLP(shared_conditioned)
        arg_logit_inputs = arg_logit_inputs.reshape((N, self.arg_depth, self.arg_size))
        arg_logits = self.generate_arg_logits(arg_logit_inputs)

        args = None
        if (choosing):
            args = self.sample_arg(arg_logits, action)

        spatial_input = self.spatial_upsampler(relational_spatial)

        embedded_action = embedded_action.unsqueeze(2).unsqueeze(3)
        embedded_action = embedded_action.reshape(embedded_action.shape + (self.spatial_size, self.spatial_size))
        spatial_input = torch.cat([spatial_input, embedded_action], axis=1)
        spatial_logits_in = self.spatial_out(spatial_input)


        spatial = None
        if (choosing):
            spatial = self.sample_spatial(spatial_logits_in)

        choice = [action, args, spatial]

        return action_logits, arg_logits, spatial_logits, hidden, value, choice

    def init_hidden(self, use_torch=True, device="cuda:0"):
        return self.convLSTM.init_hidden_state(use_torch=use_torch, device=device)

    def embed_inputs(self, inputs, net_config):
        return multi_embed(inputs, self.embeddings, self.embedding_indices)

    def sample_func(self, probs):
        return np.argmax(np.random.multinomial(probs))

    def sample_action(self, action_logits):
        action = self.sample_func(action_logits[0])
        return action

    """
        arg_logit_inputs: (N, num_args) shape
    """
    def sample_arg(self, arg_logits, action):
        arg_out = np.zeros(self.arg_depth)
        arg_types = self.action_to_nonspatial_args(action)
        for i in arg_types:
            arg_out[i] = self.sample_func(arg_logits[0,i])
        return arg_out

    def sample_spatial(self, spatial_logits, action):
        spatial_arg_out = np.zeros((self.spatial_depth, 2))
        arg_types = self.action_to_spatial_args(action)
        for i in arg_types:
            spatial_probs_flat = spatial_logits[0,i].flatten()
            arg_index = self.sample_func(spatial_probs_flat)
            spatial_arg_out[i] = np.array([int(arg_index / self.spatial_size),
                                            arg_index % self.spatial_size])

        return spatial_arg_out

    def generate_arg_logits(arg_logit_inputs):
        initial_logits = F.softmax(arg_logit_inputs) * self.valid_args
        final_logits = initial_logits / torch.sum(initial_logits, axis=-1)
        return final_logits

    def action_to_nonspatial_args(self, action):
        args = get_action_args(action)
        args = [i for i in args if not is_spatial_arg(i)]
        return args

    def action_to_spatial_args(self, action):
        args = get_action_args(action)
        args = [i for i in args if is_spatial_arg(i)]
        return args



















#############################
