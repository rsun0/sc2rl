import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Convolutional implementation of LSTM.
    Check https://pytorch.org/docs/stable/nn.html#lstm to see reference information
"""
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):

        self.input_to_input = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_input = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

        self.input_to_forget = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_forget = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

        self.input_to_gate = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_gate = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

        self.input_to_output = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.hidden_to_output = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)

    def forward(self, input, (h_0, c_0)):

        i = F.sigmoid(self.input_to_input(input) + self.hidden_to_input(h_0))
        f = F.sigmoid(self.input_to_forget(input) + self.hidden_to_forget(h_0))
        g = F.tanh(self.input_to_gate(input) + self.hidden_to_gate(h_0))
        o = F.sigmoid(self.input_to_output(input) + self.hidden_to_output(h_0))
        c_t = f * c_0 + i * g
        h_t = o * F.tanh(c_t)

        return o, (h_0, c_0)






class ResnetBlock(nn.Module):
    def __init__(self, num_features, num_layers):
        super(ResnetBlock, self).__init__()

        self.activation = nn.ReLU
        self.residuals = nn.Sequential()
        for i in range(num_layers):
            self.residuals.add_module(
                "residual" + str(i+1),
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, 1, padding=0)
                    self.activation()
                )
            )

    def forward(self, x):
        out = self.residuals(x):
        return x + out


class SelfAttentionBlock(nn.Module):

    def __init__(self, in_size, num_features):
        super(SelfAttentionBlock, self).__init__()

        self.in_size = in_size
        self.num_features = num_features

        self.QueryLayer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.Tanh(),
            nn.InstanceNorm1d(num_features)
        )

        self.KeyLayer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.Tanh(),
            nn.InstanceNorm1d(num_features)
        )

        self.ValueLayer = nn.Sequential(
            nn.Linear(in_size, num_features),
            nn.Tanh(),
            nn.InstanceNorm1d(num_features)
        )

    """
        Take in x of shape (N, H, W, D), perform attention calculations,
        return processed output of shape (N, H, W, num_features)
    """
    def forward(self, x):
        (N, H, W, D) = x.shape
        flattened = x.flatten(start_dim=1, end_dim=-2)

        Q = self.QueryLayer(flattened)
        K = self.KeyLayer(flattened)
        V = self.ValueLayer(flattened)

        numerator = torch.matmul(Q, K.permute(0,2,1))
        scaled = numerator / (torch.sqrt(self.num_features))
        attention_weights = F.softmax(scaled)
        A = torch.matmul(attention_weights, V)
        A = A.reshape((N, H, W, -1))

        return A
