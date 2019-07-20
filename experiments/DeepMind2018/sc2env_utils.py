from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType, Player
CATEGORICAL = FeatureType.CATEGORICAL
SCALAR = FeatureType.SCALAR
import torch
import torch.nn as nn



preprocess_config = {
    "minimap_features": 0 #int, number of features in minimap image
    "screen_features": 0 #int, number of features in screen image
    "player_features": 0 #int, number of features in player variable

    "num_arg_types": 0 #int, number of sets of arguments to choose from
    "arg_sizes": 0 #int, max size of a set of arguments
    "arg_mask": [0] #int array of size num_arg_types * arg_sizes, 1 if the action can pick an arg from that index, 0 otherwise
    "base_action_to_args": {} # dictionary mapping from base_action index to arg type index, number of arg candidates, 
}


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

def generate_embeddings(net_config):

    embeddings = [minimap_embeddings, screen_embeddings] = [[], []]
    input_names = ["minimap", "screen"]
    for i in range(len(input_names)):
        base = input_names[i]
        cat_indices = net_config[base + "_categorical_indices"]
        cat_sizes = net_config[base + "_categorical_size"]
        embed_size = net_config['embedding_size']

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

"""

    Performs simultaneous embeddings for each input

"""
def multi_embed(inputs, embedding_lists, embedding_indices):
    outputs = [embed(inputs[i], embedding_lists[i], embedding_indices[i]) for i in range(len(inputs))]
    return outputs
