import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, k=5):
        super(context_embedding, self).__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return F.tanh(x)


if __name__ == '__main__':
    x = torch.randint(1, 10, (1, 20)).type(torch.float)
    print(x)
    print(x.unsqueeze(0).shape)
    kernel_size = 3
    in_channels = 1
    out_channels = 1
    dilation = 1
    convlayer = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
    print(convlayer(x.unsqueeze(0)).shape)
    print(convlayer(x.unsqueeze(0)))

    __padding = (kernel_size - 1) * dilation
    print(__padding)
    x_pad = F.pad(x, (__padding, 0))
    print(x_pad)
    torch.dot(convlayer.weight[0][0], x_pad[0][0:3])+convlayer.bias[0]

    embedding = context_embedding(1, 256, 5)
    print(embedding(x.unsqueeze(0)).shape)
    print(embedding(x.unsqueeze(0)))


