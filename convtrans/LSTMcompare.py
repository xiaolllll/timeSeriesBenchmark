import numpy as np
import torch
import matplotlib.pyplot as plt
import causal_convolution_layer
import Dataloader
import math
from torch.utils.data import DataLoader

train_dataset = Dataloader.time_series_decoder_paper(96, 4500)
validation_dataset = Dataloader.time_series_decoder_paper(96, 500)
test_dataset = Dataloader.time_series_decoder_paper(96, 1000)

criterion = torch.nn.MSELoss()

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dl = DataLoader(validation_dataset, batch_size=64)
test_dl = DataLoader(test_dataset, batch_size=128)


class LSTM_Time_Series(torch.nn.Module):
    def __init__(self, input_size=2, embedding_size=256, kernel_width=9, hidden_size=512):
        super(LSTM_Time_Series, self).__init__()

        self.input_embedding = causal_convolution_layer.context_embedding(input_size, embedding_size, kernel_width)

        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)

        self.fc1 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        # concatenate observed points and time covariate
        # (B,input size + covariate size,sequence length)
        z_obs = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        print(z_obs)

        # input_embedding returns shape (B,embedding size,sequence length)
        z_obs_embedding = self.input_embedding(z_obs)

        # permute axes (B,sequence length, embedding size)
        z_obs_embedding = self.input_embedding(z_obs).permute(0, 2, 1)

        # all hidden states from lstm
        # (B,sequence length,num_directions * hidden size)
        lstm_out, _ = self.lstm(z_obs_embedding)

        # input to nn.Linear: (N,*,Hin)
        # output (N,*,Hout)
        return self.fc1(lstm_out)


criterion_LSTM = torch.nn.MSELoss()
LSTM = LSTM_Time_Series().cuda()

lr = 0.0005
optimizer = torch.optim.Adam(LSTM.parameters(), lr=lr)
epochs = 10


def Dp(y_pred, y_true, q):
    return max([q * (y_pred - y_true), (q - 1) * (y_pred - y_true)])


def Rp_num_den(y_preds, y_trues, q):
    numerator = np.sum([Dp(y_pred, y_true, q) for y_pred, y_true in zip(y_preds, y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator, denominator


def train_epoch(LSTM, train_dl, t0=96):
    LSTM.train()
    train_loss = 0
    n = 0
    for step, (x, y, _) in enumerate(train_dl):
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        output = LSTM(x, y)
        loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + 24 - 1)], y.cuda()[:, t0:])
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n


def eval_epoch(LSTM, validation_dl, t0=96):
    LSTM.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for step, (x, y, _) in enumerate(train_dl):
            x = x.cuda()
            y = y.cuda()

            output = LSTM(x, y)
            loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + 24 - 1)], y.cuda()[:, t0:])

            eval_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]

    return eval_loss / n


def test_epoch(LSTM, test_dl, t0=96):
    with torch.no_grad():
        predictions = []
        observations = []

        LSTM.eval()
        for step, (x, y, _) in enumerate(train_dl):
            x = x.cuda()
            y = y.cuda()

            output = LSTM(x, y)

            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + 24 - 1)].cpu().numpy().tolist(),
                            y.cuda()[:, t0:].cpu().numpy().tolist()):
                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp


train_epoch_loss = []
eval_epoch_loss = []
Rp_best = 10
for e, epoch in enumerate(range(epochs)):
    train_loss = []
    eval_loss = []

    l_train = train_epoch(LSTM, train_dl)
    train_loss.append(l_train)

    l_eval = eval_epoch(LSTM, validation_dl)
    eval_loss.append(l_eval)

    Rp = test_epoch(LSTM, test_dl)

    if Rp_best > Rp:
        Rp_best = Rp

    with torch.no_grad():
        print(
            "Epoch {}: Train loss={} \t Eval loss = {} \t Rp={}".format(e, np.mean(train_loss), np.mean(eval_loss), Rp))

        train_epoch_loss.append(np.mean(train_loss))
        eval_epoch_loss.append(np.mean(eval_loss))
