from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType, Player
CATEGORICAL = FeatureType.CATEGORICAL
SCALAR = FeatureType.SCALAR
import torch
import torch.nn as nn


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



for i in range(len(features)):
    name, scale, featuretype = (features[i].name, features[i].scale, features[i].type)

    dim = scale
    if (featuretype == SCALAR or (featuretype == CATEGORICAL and dim == 2)):
        val_list = np.log(val_list.clip(0) + 1)
        dim = 1
        depths = features_depth * np.ones(n_curr)
    else:

        depths = val_list + features_depth
        val_list = np.ones(n_curr)

    addition = np.stack([zeros, depths, x_coords, y_coords])
    if (i == 0):
        indices = addition
    else:
        indices = np.concatenate([indices, addition],1)

    values = np.append(values, val_list)

    features_depth += dim




def generate_embeddings(net_config):

    embeddings = [minimap_embeddings, screen_embeddings, player_embeddings] = [[], [], []]
    input_names = ["minimap", "screen"]
    for i in range(len(input_names)):
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
