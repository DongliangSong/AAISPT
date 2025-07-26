# -*- coding: utf-8 -*-
# @Time    : 2024/8/10 16:35
# @Author  : Dongliang

import os

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
from tqdm import tqdm


def to_device(data, labels, device):
    """
    Move data and labels to the specified device.
    """
    return [d.to(device) for d in data], [l.to(device) for l in labels]


def training(model, batch_size, train_dataset, val_dataset, num_epochs, scheduler, optimizer, criterion, device,
             savepath, early_stopping):
    """
    Train the model on the training set and evaluate on the validation set.

    :param model: Well-defined neural networks.
    :param train_dataset: Training tensor dataset.
    :param val_dataset: Validation tensor dataset.
    :param num_epochs: Number of epochs for training.
    :param scheduler: Learning Rate Scheduler.
    :param optimizer: Optimizer for training.
    :param criterion: Loss function.
    :param device: Training device.
    :param savepath: Path to save the trained model.
    :return: Train and validation loss.
    """
    model.to(device)
    train_losses, val_losses = [], []
    best_loss = np.Inf

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, num_train = 0, 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

        for i, data in progress_bar:
            data_list = data[::2]  # Even indices are data
            label_list = data[1::2]  # Odd indices are labels
            data_list, label_list = to_device(data_list, label_list, device)

            for train_x, train_y in zip(data_list, label_list):
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

                for val_x, val_y in zip(data_list, label_list):
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

        print(
            'Epoch : {}/{}, lr : {}, Train_loss : {:.6f}, Val_loss: {:.6f}'.format(epoch + 1, num_epochs, lr,
                                                                                   train_loss, val_loss))

    return train_losses, val_losses, best_loss


# Inference function (adapted from previous optimization)
def inference(model, test_dataset, trace_len, FWHM, batch_size, device):
    """
    Perform inference on a test dataset.

    :param model: Trained Seg_Adap_BiLSTM_BCE model
    :param test_dataset: TSDataset instance
    :param trace_len: Length of each trajectory
    :param FWHM: Full Width at Half Maximum for label conversion
    :param batch_size: Batch size for DataLoader
    :param device: Device to run inference on
    :return: (probs, locs) - Predicted probabilities and peak locations
    """
    probs, locs = [], []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing batches")

    with torch.no_grad():
        for i, data in progress_bar:
            data = data[0].to(device)  # Extract data (first element of tuple)
            pred = model(data)
            pred = pred.squeeze().cpu().numpy()

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