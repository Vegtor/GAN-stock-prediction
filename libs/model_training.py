import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first = True)
        self.gru_2 = nn.GRU(1024, 512, batch_first = True)
        self.gru_3 = nn.GRU(512, 256, batch_first = True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.gru_1(x, h0)
        out_1 = self.dropout(out_1)
        h1 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.gru_2(out_1, h1)
        out_2 = self.dropout(out_2)
        h2 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.gru_3(out_2, h2)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(4, 32, kernel_size = 5, stride = 1, padding = 'same')
        self.conv_2 = nn.Conv1d(32, 64, kernel_size = 5, stride = 1, padding = 'same')
        self.conv_3 = nn.Conv1d(64, 128, kernel_size = 3, stride = 1, padding = 'same')
        self.linear_1 = nn.Linear(128, 220)
        self.linear_2 = nn.Linear(220, 220)
        self.linear_3 = nn.Linear(220, 1)
        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_1 = self.leaky(conv_1)
        conv_2 = self.conv_2(conv_1)
        conv_2 = self.leaky(conv_2)
        conv_3 = self.conv_3(conv_2)
        conv_3 = self.leaky(conv_3)
        flatten_x = conv_3.reshape(conv_3.shape[0], conv_3.shape[1])
        out_1 = self.linear_1(flatten_x)
        out_1 = self.leaky(out_1)
        out_2 = self.linear_2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear_3(out_2)
        out = self.sigmoid(out_3)
        return out

def sliding_window(x, y, window):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan

def epoch_cycle(device, train_x, train_x_slide, test_x_slide, train_y_gan):
    batch_size = 128
    learning_rate = 0.00016
    num_epochs = 165

    trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), batch_size=batch_size, shuffle=False)

    model_G = Generator(train_x.shape[1]).to(device)
    model_D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=(0, 0.9))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=(0, 0.9))

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=165, eta_min=0, last_epoch=-1)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=165, eta_min=0, last_epoch=-1)
    for epoch in range(num_epochs):
        loss_G = []
        loss_D = []
        for (x, y) in trainDataloader:
            x = x.to(device)
            y = y.to(device)

            generated_data = model_G(x)
            generated_data = torch.cat([y[:, :4, :], generated_data.reshape(-1, 1, 1)], axis=1)

            discriminant_out_real = model_D(y)
            real_labels = torch.ones_like(discriminant_out_real).to(device)
            lossD_real = criterion(discriminant_out_real, real_labels)

            discriminant_out_gen = model_D(generated_data)
            gen_labels = torch.zeros_like(real_labels).to(device)
            lossD_gen = criterion(discriminant_out_gen, gen_labels)

            lossD = (lossD_real + lossD_gen)

            model_D.zero_grad()
            lossD.backward(retain_graph=True)
            optimizer_D.step()
            loss_D.append(lossD.item())

            output_gen = model_D(generated_data)
            lossG = criterion(output_gen, real_labels)

            model_G.zero_grad()
            lossG.backward()
            optimizer_G.step()
            loss_G.append(lossG.item())

            scheduler_1.step()
            scheduler_2.step()
    return model_G

def training(data, out_prices, param_d, param_g):
    x = data.values
    y = out_prices.values
    split_1 = int(x.shape[0] * 0.7)
    train_y = y[3:split_1 + 3, :]
    train_x = x[:split_1, :]
    test_y = y[split_1:, :]
    test_x = x[split_1:, :]

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)
    train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y = y_scaler.transform(test_y.reshape(-1, 1))

    train_x_slide, train_y_slide, train_y_gan = sliding_window(train_x, train_y, 4)
    test_x_slide, test_y_slide, test_y_gan = sliding_window(test_x, test_y, 4)

    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    model_G = epoch_cycle(device, train_x, train_x_slide, test_x_slide, train_y_gan)

    model_G.eval()
    pred_y_train = model_G(train_x_slide.to(device))
    pred_y_test = model_G(test_x_slide.to(device))

    y_train_true = y_scaler.inverse_transform(train_y_slide)
    y_train_pred = y_scaler.inverse_transform(pred_y_train.cpu().detach().numpy())

    y_test_true = y_scaler.inverse_transform(test_y_slide)
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    MSE = mean_squared_error(y_test_true, y_test_pred)
    RMSE = math.sqrt(MSE)
    generator_shape = train_x.shape[1]

    return model_G, generator_shape, MSE, RMSE

def use_model(model_file, model_shape):
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    model_load = Generator(model_shape).to(device)
    state = torch.load(model_file,map_location=torch.device('cpu'))
    model_load.load_state_dict(state)
    return model_load



