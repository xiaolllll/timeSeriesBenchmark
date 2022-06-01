__package__ = 'timeSeriesBenchmark.utils'
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.embed import DataEmbedding

from typing import List


class MLP(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 d_model: int,
                 num_hidden_dimensions: List[int],
                 hist_len: int,
                 pred_len: int,
                 freq: str = 'H',
                 use_time_feat: bool = True):
        super(MLP, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.d_model = d_model
        self.num_hidden_dimensions = num_hidden_dimensions
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.freq = freq

        # Embedding
        self.embedding = DataEmbedding(c_in, d_model, freq, use_time_feat=use_time_feat)

        modules = []
        dims = self.num_hidden_dimensions

        for i, units in enumerate(dims):
            if i == 0:
                input_size = hist_len
            else:
                input_size = dims[i - 1]
            modules += [nn.Linear(input_size, units), nn.ReLU()]

        modules.append(nn.Linear(dims[-1], pred_len))

        self.mlp = nn.Sequential(*modules)

        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x, x_mark, y_mark):

        x = self.embedding(x, x_mark)

        x = x.transpose(1, 2)

        x = self.mlp(x)

        x = x.transpose(1, 2)
        x = self.projection(x)
        return x


if __name__ == '__main__':
    hist_len = 96
    pred_len = 24
    c_in = 1  # input target feature dimension
    c_out = 1  # output target feature dimension
    d_model = 16  # model dimension
    use_time_freq = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    freq = 'H'
    num_hidden_dimension = [512, 256, 128]
    model = MLP(c_in,
            c_out,
            d_model,
            num_hidden_dimension,
            hist_len,
            pred_len,
            freq,
            use_time_freq).float().to(device)
