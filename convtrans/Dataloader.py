import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class time_series_decoder_paper(Dataset):
    """构造数据集 文章5.1"""

    def __init__(self, t0=96, N=4500, transform=None):
        """
        Args:
            t0: 根据历史数据集
            N: 数据个数
            transform: 数据转换
        """
        self.t0 = t0
        self.N = N
        self.transform = None

        # 时间点
        self.x = torch.cat(N * [torch.arange(0, t0 + 24).type(torch.float).unsqueeze(0)])

        # sin信号
        A1, A2, A3 = 60 * torch.rand(3, N)
        A4 = torch.max(A1, A2)
        self.fx = torch.cat([A1.unsqueeze(1) * torch.sin(np.pi * self.x[0, 0:12] / 6) + 72,
                             A2.unsqueeze(1) * torch.sin(np.pi * self.x[0, 12:24] / 6) + 72,
                             A3.unsqueeze(1) * torch.sin(np.pi * self.x[0, 24:t0] / 6) + 72,
                             A4.unsqueeze(1) * torch.sin(np.pi * self.x[0, t0:t0 + 24] / 12) + 72], 1)
        print(self.fx)
        print(self.fx.shape)

        # 添加噪声
        self.fx = self.fx + torch.randn(self.fx.shape)

        self.masks = self._generate_square_subsequent_mask(t0)

        print("x: {}*{}".format(*list(self.x.shape)),
              "fx: {}*{}".format(*list(self.fx.shape)))

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x[idx, :],
                  self.fx[idx, :],
                  self.masks)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_square_subsequent_mask(self, t0):
        mask = torch.zeros(t0 + 24, t0 + 24)
        for i in range(0, t0):
            mask[i, t0:] = 1
        for i in range(t0, t0 + 24):
            mask[i, i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))  # .masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':
    train_dataset = time_series_decoder_paper(96, 4500)
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for step, (x, y, _) in enumerate(train_dl):
        print(x)
        print(y)
    time = time_series_decoder_paper()
    print(time.masks)
