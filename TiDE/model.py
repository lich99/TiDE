
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class TiDE(nn.Module):
    def __init__(self, L, H, feature_dim, feature_encode_dim, decode_dim, hidden_dim, dropout, bias): 
        super().__init__()

        flatten_dim = L + (L + H) * feature_encode_dim
        self.H = H
        self.L = L
        self.decode_dim = decode_dim

        self.feature_encoder = ResBlock(feature_dim, 768, feature_encode_dim, dropout, bias)
        self.encoders = nn.Sequential(*([ ResBlock(flatten_dim, 768, hidden_dim, dropout, bias)]*1))
        self.decoders = nn.Sequential(*([ ResBlock(hidden_dim, 768, decode_dim * H, dropout, bias)]*1))
        self.final_decoder = ResBlock(decode_dim + feature_encode_dim, 768, 1, dropout, bias)
        self.residual_proj = nn.Linear(L, H, bias=bias)
        
    def forward(self, lookback, dynamic):

        feature = self.feature_encoder(dynamic)
        hidden = self.encoders(torch.cat([lookback, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.H, self.decode_dim)
        prediction = self.final_decoder(torch.cat([feature[:,self.L:], decoded], dim=-1)).squeeze(-1) + self.residual_proj(lookback)

        return prediction