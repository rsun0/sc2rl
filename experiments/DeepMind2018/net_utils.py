import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


"""
    Convolutional implementation of LSTM.
    Check https://pytorch.org/docs/stable/nn.html#lstm to see reference information
"""
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvLSTM, self).__init__()

        self.input_size, self.hidden_size = input_size, hidden_size

        self.input_to_input = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_input = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

        self.input_to_forget = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_forget = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

        self.input_to_gate = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_gate = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

        self.input_to_output = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_output = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

    """
        input: torch tensor of shape (N, D, H, W)
        hidden_state: torch tensor of shape (N, 2, D, H, W)

        D is placeholder as usual
    """
    def forward(self, input, hidden_state):

        h_0 = hidden_state[:,0]
        c_0 = hidden_state[:,1]
        i = F.sigmoid(self.input_to_input(input) + self.hidden_to_input(h_0))
        f = F.sigmoid(self.input_to_forget(input) + self.hidden_to_forget(h_0))
        g = F.tanh(self.input_to_gate(input) + self.hidden_to_gate(h_0))
        o = F.sigmoid(self.input_to_output(input) + self.hidden_to_output(h_0))
        c_t = f * c_0 + i * g
        h_t = o * F.tanh(c_t)

        hidden_state_out = torch.cat([h_t.unsqueeze(1), c_t.unsqueeze(1)], dim=1)

        return o, hidden_state_out

    def init_hidden_state(self, batch_size=1, width=8, use_torch=True, device="cuda:0"):
        if (use_torch):
            return torch.zeros((batch_size, 2, self.hidden_size, width, width)).float().to(device)
        else:
            return np.zeros((batch_size, 2, self.hidden_size, width, width)).astype(np.float32)




class ResnetBlock(nn.Module):
    def __init__(self, num_features, num_layers):
        super(ResnetBlock, self).__init__()

        self.activation = nn.ReLU
        self.residuals = nn.Sequential()
        for i in range(num_layers):
            self.residuals.add_module(
                "residual" + str(i+1),
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                    self.activation()
                )
            )

    def forward(self, x):
        out = self.residuals(x)
        return x + out

class RelationalProjection(nn.Module):
    def __init__(self, in_size, num_features):

        super(RelationalProjection, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.InstanceNorm1d(num_features)
        )

    def forward(self, x):
        return self.layer(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_size, num_features, num_heads, device="cuda:0"):

        super(SelfAttentionBlock, self).__init__()
        self.device = device
        self.in_size = in_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.QueryLayers = []
        self.KeyLayers = []
        self.ValueLayers = []

        for i in range(in_size):
            self.QueryLayers.append(RelationalProjection(in_size, num_features).to(self.device))
            self.KeyLayers.append(RelationalProjection(in_size, num_features).to(self.device))
            self.ValueLayers.append(RelationalProjection(in_size, num_features).to(self.device))
        self.MLP = nn.Sequential(
            nn.Linear(self.num_heads * self.num_features, self.num_heads * self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_heads * self.num_features, self.num_heads * self.num_features),
            nn.ReLU()
        ).to(self.device)

        self.output_norm = nn.InstanceNorm1d(num_features)

    def forward(self, x):
        (N, D, H, W) = x.shape
        new_x = x.permute(0, 2, 3, 1)
        flattened = new_x.flatten(start_dim=1, end_dim=-2)

        heads_out = []

        for i in range(self.num_heads):
            Q = self.QueryLayers[i](flattened)
            K = self.KeyLayers[i](flattened)
            V = self.ValueLayers[i](flattened)
            numerator = torch.matmul(Q, K.permute(0,2,1))
            scaled = numerator / math.sqrt(self.num_features)
            attention_weights = F.softmax(scaled)
            A = torch.matmul(attention_weights, V)
            heads_out.append(A)

        heads_out = torch.cat(heads_out, dim=-1)
        output = self.output_norm(self.MLP(heads_out) + heads_out)
        output = output.permute(0,2,1).contiguous().view((N, -1, H, W))

        return output



class RelationalModule(nn.Module):

    def __init__(self, in_size, num_features, num_heads):
        super(SelfAttentionBlock, self).__init__()

        self.in_size = in_size
        self.num_features = num_features

        self.QueryLayer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.InstanceNorm1d(num_features)
        )

        self.KeyLayer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.InstanceNorm1d(num_features)
        )

        self.ValueLayer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.InstanceNorm1d(num_features)
        )

        self.QueryNorm = nn.InstanceNorm1d(num_features)
        self.KeyNorm = nn.InstanceNorm1d(num_features)
        self.ValueNorm = nn.InstanceNorm1d(num_features)

    """
        Take in x of shape (N, D, H, W), perform attention calculations,
        return processed output of shape (N, num_features, H, W)
    """
    def forward(self, x):
        (N, D, H, W) = x.shape
        new_x = x.permute(0, 2, 3, 1)
        flattened = new_x.flatten(start_dim=1, end_dim=-2)

        Q = self.QueryNorm(self.QueryLayer(flattened))
        K = self.KeyNorm(self.KeyLayer(flattened))
        V = self.ValueNorm(self.ValueLayer(flattened))

        numerator = torch.matmul(Q, K.permute(0,2,1))
        scaled = numerator / math.sqrt(self.num_features)
        attention_weights = F.softmax(scaled)
        A = torch.matmul(attention_weights, V)
        A_out = A.view((N, -1, H, W))

        return A_out

class Downsampler(nn.Module):
    def __init__(self, input_features, net_config):
        super(Downsampler, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_features,
                net_config['down_conv_features'],
                kernel_size=(4,4),
                stride=2,
                padding=1),
            nn.ReLU(),

            ResnetBlock(net_config['down_conv_features'],
                            net_config['relational_depth']),
            ResnetBlock(net_config['down_conv_features'],
                            net_config['relational_depth'])
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(net_config['down_conv_features'],
                2*net_config['down_conv_features'],
                kernel_size=(4,4),
                stride=2,
                padding=1),
            nn.ReLU(),

            ResnetBlock(2*net_config['down_conv_features'],
                            net_config['relational_depth']),
            ResnetBlock(2*net_config['down_conv_features'],
                            net_config['relational_depth'])
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(2*net_config['down_conv_features'],
                4*net_config['down_conv_features'],
                kernel_size=(4,4),
                stride=2,
                padding=1),
            nn.ReLU(),

            ResnetBlock(4*net_config['down_conv_features'],
                            net_config['relational_depth']),
            ResnetBlock(4*net_config['down_conv_features'],
                            net_config['relational_depth'])
        )

    def forward(self, input):
        h1 = self.block1(input)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        return h3

class SpatialUpsampler(nn.Module):
    def __init__(self, net_config, output_depth):
        super(SpatialUpsampler, self).__init__()
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                net_config['relational_features'] * net_config["relational_heads"] + net_config["action_embedding_size"],
                net_config['up_conv_features'],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                net_config['up_conv_features'],
                int(0.5 * net_config['up_conv_features']),
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(int(0.5 * net_config['up_conv_features']),
                                        output_depth,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0
            ),
            nn.ReLU()
        )

    def forward(self, x):

        h1 = self.tconv1(x)
        h2 = self.tconv2(h1)
        h3 = self.conv_out(h2)

        return h3

class FastEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device="cuda:0"):
        super(FastEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.from_numpy(np.random.uniform(-1,1, (num_embeddings, embedding_dim))).float().to(device)

    def forward(self, x):
        s = x.shape
        values = self.weight[x.view(-1)]
        return values.view(s + (self.embedding_dim,))


class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-1)

class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)
