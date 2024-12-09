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
    x_new = []
    new = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_new.append(tmp_x)
        new.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_new = torch.from_numpy(np.array(x_new)).float()
    new = torch.from_numpy(np.array(new)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_new, new, y_gan

def models_preparation_gan(data, out_prices):
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

    return {'x_scaler': x_scaler, 'train_x': train_x, 'test_x': test_x, 'train_x_slide': train_x_slide, 'test_x_slide': test_x_slide,
            'y_scaler': y_scaler, 'train_y': train_y, 'test_y': test_y, 'train_y_slide': train_y_slide, 'test_y_slide': test_y_slide,
            'train_y_gan': train_y_gan, 'test_y_gan': test_y_gan}

def setup_training_gan(prepared_data, batch_size=128, learning_rate=0.00016, betas_G=(0, 0.9), betas_D=(0, 0.9)):
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    trainDataloader = DataLoader(TensorDataset(prepared_data['train_x_slide'], prepared_data['train_y_gan']), batch_size=batch_size, shuffle=False)

    model_G = Generator(prepared_data['train_x'].shape[1]).to(device)
    model_D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=betas_G)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=betas_D)

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=165, eta_min=0, last_epoch=-1)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=165, eta_min=0, last_epoch=-1)

    return {
        'device': device, 'criterion': criterion, 'data_loader': trainDataloader,
        'model_G': model_G, 'optimizer_G': optimizer_G, 'scheduler_G': scheduler_1,
        'model_D': model_D, 'optimizer_D': optimizer_D, 'scheduler_D': scheduler_2,
    }

def run_model_gan(prepared_models, num_epochs=165):
    device = prepared_models['device']
    criterion = prepared_models['criterion']
    train_loader = prepared_models['train_loader']

    model_G = prepared_models['model_G']
    optimizer_G = prepared_models['optimizer_G']
    scheduler_G = prepared_models['scheduler_G']

    model_D = prepared_models['model_D']
    optimizer_D = prepared_models['optimizer_D']
    scheduler_D = prepared_models['scheduler_D']

    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Generator forward pass
            generated_data = model_G(x)
            generated_data = torch.cat([y[:, :4, :], generated_data.reshape(-1, 1, 1)], axis=1)

            # Discriminator training
            real_labels = torch.ones_like(model_D(y)).to(device)
            gen_labels = torch.zeros_like(real_labels).to(device)

            loss_D_real = criterion(model_D(y), real_labels)
            lossD_gen = criterion(model_D(generated_data), gen_labels)
            loss_D = loss_D_real + lossD_gen

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # Generator training
            output_gen = model_D(generated_data)
            loss_G = criterion(output_gen, real_labels)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Update learning rates
            scheduler_G.step()
            scheduler_D.step()
    return model_G

def evaluate_model_gan(model_G, prepared_data):
    device = prepared_data['device']
    y_scaler = prepared_data['y_scaler']

    model_G.eval()
    pred_y_train = model_G(prepared_data['train_x_slide'].to(device))
    pred_y_test = model_G(prepared_data['test_x_slide'].to(device))

    y_train_true = y_scaler.inverse_transform(prepared_data['train_y_slide'])
    y_train_pred = y_scaler.inverse_transform(pred_y_train.cpu().detach().numpy())

    y_test_true = y_scaler.inverse_transform(prepared_data['test_y_slide'])
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    MSE = mean_squared_error(y_test_true, y_test_pred)
    RMSE = math.sqrt(MSE)
    generator_shape = prepared_data['train_x'].shape[1]

    return {
        'mse': MSE, 'rmse': RMSE, 'generator_shape': generator_shape,
        'train_true': y_train_true, 'train_pred': y_train_pred,
        'test_true': y_test_true, 'test_pred': y_test_pred
    }
def train_cycle_gan(data, out_prices, batch_size=128, learning_rate=0.00016, betas_G=(0, 0.9), betas_D=(0, 0.9), num_epochs=165):
    prepared_data = models_preparation_gan(data, out_prices)
    prepared_models = setup_training_gan(prepared_data, batch_size, learning_rate, betas_G, betas_D)
    model_G = run_model_gan(prepared_models, num_epochs)
    results = evaluate_model_gan(model_G, prepared_data)
    return results

def use_model(model_file, model_shape, use_cuda=1):
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    model_load = Generator(model_shape).to(device)
    state = torch.load(model_file,map_location=torch.device('cpu'))
    model_load.load_state_dict(state)
    return model_load



