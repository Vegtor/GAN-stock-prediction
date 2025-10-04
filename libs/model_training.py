import numpy as np
import torch
import torch.nn as nn
from sympy import residue
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math


class Generator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru_1 = nn.GRU(input_size, 1024, batch_first=True)
        self.gru_2 = nn.GRU(1024, 512, batch_first=True)
        self.gru_3 = nn.GRU(512, 256, batch_first=True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        device = x.device
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
        self.conv_1 = nn.Conv1d(5, 32, kernel_size=5, stride=1, padding='same')
        self.conv_2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding='same')
        self.conv_3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same')
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


def sliding_window(x, y, window=4):
    """
    Function for reshaping data into small sections of data - windows.

    Args:
        x (pandas.DataFrame): DataFrame containing features.
        y (list): List of target variables.
        window (int, optional): Size of the sliding window.
            Defaults to 4.

    Returns:
        A tuple containing a 3D array (array with 3 dimensions containing features in structures of windows),
        an array (array of target values for specific windows)
        and an array (Array of target values to be used in structure of windows)
    """
    x_new = []
    y_new = []
    y_gan = []
    for i in range(window, x.shape[0]):
        tmp_x = x[i - window: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window: i + 1]
        x_new.append(tmp_x)
        y_new.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_new = torch.from_numpy(np.array(x_new)).float()
    y_new = torch.from_numpy(np.array(y_new)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_new, y_new, y_gan


def data_preparation_gan(data, target):
    """
    Preparation of data and MinMax scaling. Separation to test and train, creation of learning windows.

    Args:
        data (pandas.DataFrame): DataFrame containing features.
        target (list): List of target variables.

    Returns:
        A dictionary containing MinMax scalers and data seperated to test and train parts,
        together with versions structured to windows.
        Precise list of return variables is: x_scaler, train_x, test_x, train_x_slide,
        test_x_slide, y_scaler, train_y, test_y, train_y_slide, test_y_slide, train_y_gan, test_y_gan.
        Variables with slide in name correspond to versions with windows.
    """
    x = data.values
    y = target.values
    split_1 = int(x.shape[0] * 0.7)
    train_y = y[3:split_1 + 3]
    train_x = x[:split_1, :]
    test_y = y[split_1:]
    test_x = x[split_1:, :]

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)
    train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y = y_scaler.transform(test_y.reshape(-1, 1))

    train_x_slide, train_y_slide, train_y_gan = sliding_window(train_x, train_y, 4)

    test_x_slide, test_y_slide, test_y_gan = sliding_window(test_x, test_y, 4)

    return {'x_scaler': x_scaler, 'train_x': train_x, 'test_x': test_x, 'train_x_slide': train_x_slide,
            'test_x_slide': test_x_slide,
            'y_scaler': y_scaler, 'train_y': train_y, 'test_y': test_y, 'train_y_slide': train_y_slide,
            'test_y_slide': test_y_slide,
            'train_y_gan': train_y_gan, 'test_y_gan': test_y_gan}


def setup_training_gan(prepared_data, batch_size=128, learning_rate=0.00016,
                       betas_G=(0.0, 0.9), betas_D=(0.0, 0.9), tmax_G=165, tmax_D=165):
    """
    Setup models, optimizers and schedulers for training of GAN based on set parameters.

    Args:
        prepared_data (dictionary): Dictionary containing prepared data. It is an output of models_preparation_gan.
        batch_size (int, optional): Size of mini-batch.
            Defaults to 128.
        learning_rate (float, optional): Learning rate.
            Defaults to 0.00016.
        betas_G (tuple, optional): Beta parameters of optimizer for generator.
            Defaults to (0, 0.9).
        betas_D (tuple, optional): Beta parameters of optimizer for discriminator.
            Defaults to (0, 0.9).
        tmax_G (int, optional): Max number of training iterations for cosine annealing scheduler for generator.
            Defaults to 165.
        tmax_D (int, optional): Max number of training iterations for cosine annealing scheduler for discriminator.
            Defaults to 165.

    Returns:
        A dictionary containing pytorch device, function for learning criteria, instance of DataLoader on training data
        and there is model, optimizer and scheduler for generator and discriminator (respectively).
    """
    use_cuda = 1
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda)  else "cpu")
    trainDataloader = DataLoader(TensorDataset(prepared_data['train_x_slide'],
                                               prepared_data['train_y_gan']), batch_size=batch_size, shuffle=False)

    model_G = Generator(prepared_data['train_x'].shape[1])
    model_G = model_G.to(device)
    model_D = Discriminator()
    model_D = model_D.to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=learning_rate, betas=betas_G)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=learning_rate, betas=betas_D)

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=tmax_G, eta_min=0, last_epoch=-1)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=tmax_D, eta_min=0, last_epoch=-1)

    return {
        'device': device, 'criterion': criterion, 'data_loader': trainDataloader,
        'model_G': model_G, 'optimizer_G': optimizer_G, 'scheduler_G': scheduler_G,
        'model_D': model_D, 'optimizer_D': optimizer_D, 'scheduler_D': scheduler_D,
    }


def train_gan(prepared_models, num_epochs=165):
    """
    Training cycle for GAN model with set number of epochs.

    Args:
        prepared_models (dictionary): Dictionary containing prepared models. It is an output of setup_training_gan.
        num_epochs (int, optional): Number of training epochs.
            Defaults to 165.

    Returns:
        Model for generating data.
    """
    device = prepared_models['device']
    criterion = prepared_models['criterion']
    train_loader = prepared_models['data_loader']

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
            '''
            with torch.no_grad():
            label_shape = model_D(y).shape
        
            real_labels = torch.ones(label_shape, device=device)
            gen_labels = torch.zeros(label_shape, device=device)
            '''

            # Labels
            real_labels = torch.ones_like(model_D(y)).to(device)
            gen_labels = torch.zeros_like(real_labels).to(device)

            test = model_D(y)

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


def sample_noise_from_residuals(residuals, size=1):
    return np.random.choice(residuals, size=size, replace=True)


def evaluate_gan(model_G, prepared_data, device, noise=0):
    """
    Evaluation of model from GAN training phase.

    Args:
        model_G(Generator): Model of generator from GAN training process.
        prepared_data (dictionary): Dictionary containing prepared data. It is an output of models_preparation_gan.
        device (torch.device): Device used for whole process of training.
        noise (bool): Add noise for robust testing.
            Defaults to 0.

    Returns:
        A dictionary containing MSE of generator, RMSE of generator, generator shape,
        training and testing values with predictions based on model.
    """
    y_scaler = prepared_data['y_scaler']


    model_G.eval()
    with torch.no_grad():
        pred_y_train = model_G(prepared_data['train_x_slide'].to(device))
        pred_y_test = model_G(prepared_data['test_x_slide'].to(device))

    y_train_true = y_scaler.inverse_transform(prepared_data['train_y_slide'])
    y_train_pred = y_scaler.inverse_transform(pred_y_train.cpu().detach().numpy())

    y_test_true = y_scaler.inverse_transform(prepared_data['test_y_slide'])
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    if noise:
        residues = y_train_true - y_train_pred
        residues = residues.flatten()
        noise_values = sample_noise_from_residuals(residues, len(y_test_true))
        y_test_true = y_test_true.flatten() + noise_values

    MSE = mean_squared_error(y_test_true, y_test_pred)
    RMSE = math.sqrt(MSE)
    generator_shape = prepared_data['train_x'].shape[1]

    return {
        'mse': MSE, 'rmse': RMSE, 'generator_shape': generator_shape,
        'train_true': y_train_true, 'train_pred': y_train_pred,
        'test_true': y_test_true, 'test_pred': y_test_pred
    }


def train_process_gan(data, target, num_epochs=165, batch_size=128, learning_rate=0.00016, betas_G=(0.0, 0.9),
                      betas_D=(0.0, 0.9), tmax_G=165, tmax_D=165):
    """
    Whole process of training cycle for GAN model with set parameters.

    Args:
        data (pandas.DataFrame): DataFrame containing features.
        target (list): List of target variables.
        batch_size (int, optional): Size of mini-batch.
            Defaults to 128.
        num_epochs (int, optional): Number of training epochs.
            Defaults to 165.
        learning_rate (float, optional): Learning rate.
            Defaults to 0.00016.
        betas_G (tuple, optional): Beta parameters of optimizer for generator.
            Defaults to (0, 0.9).
        betas_D (tuple, optional): Beta parameters of optimizer for discriminator.
            Defaults to (0, 0.9).
        tmax_G (int, optional): Max number of training iterations for cosine annealing scheduler for generator.
            Defaults to 165.
        tmax_D (int, optional): Max number of training iterations for cosine annealing scheduler for discriminator.
            Defaults to 165.

    Returns:
        Same return as function evaluate_model_gan. A dictionary containing MSE of generator, RMSE of generator,
        generator shape, training and testing values with predictions based on model.
    """
    prepared_data = data_preparation_gan(data, target)
    prepared_models = setup_training_gan(prepared_data, batch_size, learning_rate, betas_G, betas_D, tmax_G, tmax_D)
    model_G = train_gan(prepared_models, num_epochs)
    results = evaluate_gan(model_G, prepared_data, prepared_models["device"])
    return {
        "results": results,
        "model": model_G
    }


def load_gan(model_file, model_shape, use_cuda=1):
    """
    Loading of Torch model from state dict file.

    Args:
        model_file (string): Path to file containing Torch state dict of generator.
        model_shape (int): Number of features used for training.
        use_cuda (int, optional): True or false statement in int format about using cuda.
            Defaults to 1.

    Returns:
        A dictionary with loaded Torch model of generator and set device.
    """
    device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
    model_load = Generator(model_shape).to(device)
    state = torch.load(model_file, map_location=torch.device('cpu'))
    model_load.load_state_dict(state)
    return {'model': model_load, 'device': device}

def output_from_model(model_path, data, target, noise=False):
    data_result = data_preparation_gan(data, target)
    model_path = model_path
    model = load_gan(model_path, data_result["test_x_slide"].size(dim=2))
    results = evaluate_gan(model["model"], data_result, model["device"],noise)
    return results
