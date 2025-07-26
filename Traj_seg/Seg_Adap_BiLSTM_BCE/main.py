# -*- coding: utf-8 -*-
# @Time    : 2024/12/10 15:57
# @Author  : Dongliang

import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.signal import find_peaks
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from AAISPT.pytorch_tools import EarlyStopping

# Set GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TSDataset(Dataset):
    def __init__(self, data_dict, label_dict):
        super(TSDataset, self).__init__()
        self.data = {key: np.array(val, dtype=np.float32) for key, val in data_dict.items()}
        self.labels = {key: np.array(val, dtype=np.float32) for key, val in label_dict.items()}

    def __getitem__(self, idx):
        result = []
        for data_key, label_key in zip(self.data.keys(), self.labels.keys()):
            result.append(torch.tensor(self.data[data_key][idx], dtype=torch.float32))
            result.append(torch.tensor(self.labels[label_key][idx], dtype=torch.float32))
        return tuple(result)

    def __len__(self):
        return len(next(iter(self.data.values())))


def label_convert(label, seq_len, FWHM):
    """
    Convert each trajectory's change point to a Gaussian distribution with a specific half-peak width.

    :param label: The change point of each track.
    :param seq_len: The length of the trajectory.
    :param FWHM: The full-width at half-maximum of the Gaussian distribution.
    :return: The transformed labelling matrix.
    """

    def gaussian(x, mu, sigma):
        return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    x = np.linspace(0, seq_len - 1, seq_len)[np.newaxis, :]  # Shape: (1, seq_len)
    labels = np.array(label)[:, np.newaxis]  # Shape: (num_labels, 1)
    y = gaussian(x, labels, sigma)
    return y


class AdaptiveConvModule(nn.Module):
    def __init__(self, max_dim, target_dim=64):
        super(AdaptiveConvModule, self).__init__()

        # Create corresponding 1x1 convolutional layers for each input dimension.
        self.dim_conv = nn.ModuleDict({
            f'dim_{d}': nn.Conv1d(d, target_dim, kernel_size=1)
            for d in range(2, max_dim + 1)
        })
        self.target_dim = target_dim
        self.norm = nn.LayerNorm(target_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        conv = self.dim_conv[f'dim_{input_dim}']
        x = x.transpose(1, 2)
        x = conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Seg_Adap_BiLSTM_BCE(nn.Module):
    def __init__(self, max_dim, d_model, hidden_size, num_layers, out_dim):
        super(Seg_Adap_BiLSTM_BCE, self).__init__()
        self.conv_block = AdaptiveConvModule(max_dim=max_dim, target_dim=d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_block(x)
        lstm_out, _ = self.lstm(x)
        fc_out = self.fc(lstm_out)
        outputs = self.sigmoid(fc_out)
        return outputs


def to_device(data, labels, device):
    """
    Move data and labels to the specified device.
    """
    return [data.to(device) for data in data], [label.to(device) for label in labels]


def training(model, train_dataset, val_dataset, num_epochs, optimizer, criterion, device, savepath, batch_size,
             early_stopping):
    """
    Train the model on the training set and evaluate on the validation set.

    :param model: Well-defined neural networks.
    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param num_epochs: Number of epochs for training.
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param device: Training device.
    :param savepath: Path to save the trained model.
    :param batch_size: The number of samples per batch used for training.
    :param early_stopping: Create an early stopping object.
    :return: A tuple containing:
            - List of average training losses per epoch.
            - List of average validation losses per epoch.
            - Best validation loss achieved during training.
    """

    model.to(device)
    train_losses, val_losses = [], []
    best_loss = np.Inf

    for epoch in range(num_epochs):
        model.train()
        train_loss, num_train = 0, 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for i, data in progress_bar:
            data_list = data[::2]
            label_list = data[1::2]
            data_list, label_list = to_device(data_list, label_list, device)

            for train_x, train_y in zip(data_list,label_list):
                output = model(train_x)
                loss = criterion(output.float(), train_y.float())
                num_train += len(train_y)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_loss /= num_train
        train_losses.append(train_loss)

        # Validation
        val_loss, num_val = 0, 0
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

            for i, data in progress_bar:
                data_list = data[::2]
                label_list = data[1::2]
                data_list, label_list = to_device(data_list, label_list, device)

                for val_x, val_y in zip(data_list,label_list):
                    output = model(val_x)
                    loss = criterion(output.float(), val_y.float())
                    num_val += len(val_y)
                    val_loss += loss.item()

        val_loss /= num_val
        val_losses.append(val_loss)

        # update lr if val_loss increase beyond patience
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(savepath, 'best_model.pth'))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f'Epoch [{epoch + 1}/{num_epochs}], lr: {lr:.6f}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return train_losses, val_losses, best_loss


def inference(model, test_X, test_Y, FWHM, batch_size, device='cuda'):
    """
    Perform inference on a trajectory segmentation model using a test dataset.
    This function processes the test data through the provided model, applies label conversion,
    performs batch-wise inference, and detects peaks in the predicted outputs within a region
    of interest (ROI). It returns the predicted probabilities and peak locations.

    :param model: The trained PyTorch model (e.g., Seg_Adap_BiLSTM_BCE) for trajectory segmentation.
    :param test_X: Test input data of shape (nums, trace_len, dim), where nums is the number of samples,
        trace_len is the length of each trajectory, and dim is the dimensionality.
    :param test_Y: Test labels corresponding to test_X, used for label conversion.
    :param FWHM: Full Width at Half Maximum for label conversion.
    :param batch_size: Batch size for the DataLoader.
    :param device: Device to run the model on ('cuda' or 'cpu', default: 'cuda').
    :return:
        probs: List of predicted probabilities for each sample, where each element is a numpy array
        of shape (trace_len,) containing the model's output probabilities.
        locs: List of detected peak locations for each sample. Each location is an integer
        representing the index of the maximum peak in the ROI (adjusted for offset).
        If no peak is found, -1 is returned for that sample.
    """

    # Load test dataset
    nums, trace_len, dim = test_X.shape

    # Label conversion
    test_Y = label_convert(test_Y, trace_len, FWHM=FWHM)
    test_Y = test_Y[:, :, np.newaxis]
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.long)
    test_dataset = TSDataset({'0': test_X}, {'0': test_Y})

    # Inference
    model.to(device)
    model.eval()
    probs, locs = [], []
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches")
        for i, data in progress_bar:
            data = data[0].to(device)
            pred = model(data)
            pred = pred.squeeze().cpu().detach().numpy()

            # Define region of interest (ROI) for peak detection
            x = np.linspace(1, trace_len, trace_len)
            x_min, x_max = 10, trace_len - 10
            mask = (x >= x_min) & (x <= x_max)
            y_roi = pred[:, mask]

            # Peak detection
            loc = []
            for i in range(len(y_roi)):
                peaks, _ = find_peaks(y_roi[i])
                if len(peaks) > 0:
                    max_peak_idx = peaks[np.argmax(y_roi[i][peaks])]
                    loc.append(max_peak_idx + 10)  # Adjust for ROI offset
                else:
                    loc.append(-1)

            probs.extend(pred)
            locs.extend(loc)

    return probs, locs


if __name__ == '__main__':
    FWHM = 10
    num_epochs = 100
    batch_size = 512
    init_lr = 1e-3
    weight_decay = 1e-4
    patience = 10  # early stopping patience

    max_dim = 5
    d_model = 128
    hidden_size = 128
    num_layers = 3
    out_dim = 1
    trace_len = 200

    root = Path('../data')
    savepath = root / 'Seg_Adap_BiLSTM_BCE'
    savepath.mkdir(parents=True, exist_ok=True)

    # Dynamically discover dimension directories
    dims = [d.name for d in root.iterdir() if d.is_dir() and d.name.endswith('D')]
    dims.sort()  # Ensure consistent order (e.g., 2D, 3D, 4D, ...)
    paths = [root / dim for dim in dims]
    print(f"Found dimensions: {dims}")

    data_dict_train, label_dict_train = {}, {}
    data_dict_val, label_dict_val = {}, {}

    for i, (dim, path) in enumerate(zip(dims, paths)):
        train = scipy.io.loadmat(os.path.join(path, 'norm_train.mat'))
        val = scipy.io.loadmat(os.path.join(path, 'norm_val.mat'))

        train_x, train_y = train['data'].squeeze(), train['label'].squeeze()
        val_x, val_y = val['data'].squeeze(), val['label'].squeeze()

        train_y = label_convert(train_y, trace_len, FWHM=FWHM)
        val_y = label_convert(val_y, trace_len, FWHM=FWHM)

        train_y = train_y[:, :, np.newaxis]
        val_y = val_y[:, :, np.newaxis]

        data_dict_train[dim.lower()] = train_x
        label_dict_train[dim.lower()] = train_y
        data_dict_val[dim.lower()] = val_x
        label_dict_val[dim.lower()] = val_y
        print(f'{dim} data loaded')

    train_dataset = TSDataset(data_dict_train, label_dict_train)
    val_dataset = TSDataset(data_dict_val, label_dict_val)

    # Create model, loss function, and optimizer
    model = Seg_Adap_BiLSTM_BCE(
        max_dim=max_dim,
        d_model=d_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        out_dim=out_dim
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, eps=1e-06)

    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience, verbose=True)
    train_losses, val_losses, best_loss = training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        savepath=savepath,
        batch_size=batch_size,
        early_stopping=early_stopping
    )

    # Save final model and training history
    torch.save(model.state_dict(), os.path.join(savepath, 'seg_model.pth'))
    with open(os.path.join(savepath, 'history.pkl'), 'wb') as f:
        pickle.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(num_epochs), train_losses, label='train_loss')
    plt.plot(np.arange(num_epochs), val_losses, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(savepath, 'train_and_val.png'), dpi=600)
    plt.show()

    # Saving training parameters
    params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'init_lr': init_lr,
        'weight_decay': weight_decay,
        'max_dim': max_dim,
        'd_model': d_model,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'out_dim': out_dim,
        'savepath': savepath
    }
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = os.path.join(savepath, f'params_{current_time}.txt')
    with open(file_name, 'w') as f:
        f.write('Seg_Adap_BiLSTM_BCE train parameters:\n')
        for key, value in params.items():
            f.write(f"{key} = {value}\n")
