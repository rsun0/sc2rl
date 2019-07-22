from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType, Player
CATEGORICAL = FeatureType.CATEGORICAL
SCALAR = FeatureType.SCALAR
from pysc2.lib.actions import FUNCTIONS, TYPES
import torch
import torch.nn as nn
import numpy as np

def get_action_args(action):
    base_action_func = FUNCTIONS._func_list[action]
    arg_types = base_action_func.args
    arg_ids = np.array([arg_types[i].id for i in range(len(arg_types))])
    return arg_ids

def is_spatial_arg(action_id):
    return action_id < 3

valid_args = np.zeros((1, env_config["arg_depth"], env_config["max_arg_size"]))
for i in range(10):
    type = TYPES[i+3]
    size = type.sizes[0]
    valid_args[0,i,:size] = 1

def categorical_mask(features):
    categorical_indices = []
    categorical_sizes = []
    for i in range(len(features)):
        name, scale, featuretype = (features[i].name, features[i].scale, features[i].type)
        if (featuretype == CATEGORICAL):
            categorical_indices.append(i)
            categorical_sizes.append(scale)
    return categorical_indices, categorical_sizes


minimap_categorical_indices, minimap_categorical_sizes = categorical_mask(MINIMAP_FEATURES)
screen_categorical_indices, screen_categorical_sizes = categorical_mask(SCREEN_FEATURES)




env_config = {
    "raw_minimap": len(SCREEN_FEATURES), #int, number of features in minimap image
    "raw_screen": len(MINIMAP_FEATURES), #int, number of features in screen image
    "raw_player": len(Player), #int, number of features in player variable

    "screen_categorical_indices": screen_categorical_indices,
    "minimap_categorical_indices": minimap_categorical_indices,
    "screen_categorical_size": screen_categorical_sizes,
    "minimap_categorical_size": minimap_categorical_sizes,

    "action_space": len(FUNCTIONS)
    "num_arg_types": 13, #int, number of sets of arguments to choose from
    "arg_depth": 10,
    "max_arg_size": 500,
    "spatial_action_depth": 3,
    "spatial_action_size": 84
}

def generate_embeddings(config):

    embeddings = [minimap_embeddings, screen_embeddings] = [[], []]
    input_names = ["minimap", "screen"]
    for i in range(len(input_names)):
        base = input_names[i]
        cat_indices = env_config[base + "_categorical_indices"]
        cat_sizes = env_config[base + "_categorical_size"]
        embed_size = config['state_embedding_size']

        for j in range(len(cat_indices)):
            embeddings[i].append(nn.Embedding(cat_sizes[i], embed_size))

    return embeddings

"""
    Performs log transform for scalar features
    Performs torch.nn.Embedding for categorical features
"""
def embed(x, embedding_list, embedding_indices):

    s = x.shape
    feature_dim = s[-1]
    embedding_size = embedding_list[0].embedding_dim
    output_dim = feature_dim + sum([e.embedding_dim for e in embedding_list]) - len(embedding_list)

    output = torch.zeros(s[:-1] + (output_dim,)).astype(x.dtype).to(x.device)

    lower = 0
    upper = 0
    embed_count = 0
    for i in range(len(output_dim)):
        if (i in embedding_indices):
            upper += embedding_size
            output[...,lower:upper] = embedding_list[embed_count](x[...,i])
            embed_count += 1
        else:
            upper += 1
            output[...,lower:upper] = torch.log(x[...,i]+1.0)
        lower = upper

    return output

def processed_feature_dim(feature_size, embedding_list):
    return feature_size + sum([e.embedding_dim for e in embedding_list]) - len(embedding_list)

"""

    Performs simultaneous embeddings for each input

"""
def multi_embed(inputs, embedding_lists, embedding_indices):
    outputs = [embed(inputs[i], embedding_lists[i], embedding_indices[i]) for i in range(len(inputs))]
    return outputs
