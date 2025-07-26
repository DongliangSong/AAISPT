# -*- coding: utf-8 -*-
# @Time    : 2025/7/25 17:41
# @Author  : Dongliang


import os

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_dataloaders, val_dataloaders, num_epochs, optimizer, scheduler, criterion, device,
                savepath, early_stopping):
    """
    Train and validate a multitask Transformer model, computing losses and accuracies for training and validation.

    :param model: The multitask Transformer model to train.
    :param train_dataloaders: DataLoader(s) for the training dataset(s).
    :param val_dataloaders: DataLoader(s) for the validation dataset(s).
    :param num_epochs: Number of epochs to train the model.
    :param optimizer: Optimizer for updating model parameters.
    :param scheduler: Learning rate scheduler for adjusting the learning rate.
    :param criterion: Loss function(s) for computing the loss.
    :param device: Device to run the model on (e.g., 'cuda' or 'cpu').
    :param savepath: Path to save the trained model.
    :param early_stopping: Early stopping mechanism to prevent overfitting.

    :return: A dictionary containing training and validation metrics:
        - 'total_train_losses': Average training loss per epoch.
        - 'total_val_losses': Average validation loss per epoch.
        - 'total_train_accuracies': Average training accuracy per epoch across tasks.
        - 'total_val_accuracies': Average validation accuracy per epoch across tasks.
        - 'train_task_losses': Per-task training losses for each epoch.
        - 'val_task_losses': Per-task validation losses for each epoch.
    """

    model.to(device)
    num_tasks = len(train_dataloaders)

    total_train_losses, total_val_losses = [], []
    total_train_accuracies, total_val_accuracies = [], []
    train_task_losses = []
    val_task_losses = []
    best_acc = 0

    for epoch in range(num_epochs):
        # Model training
        model.train()
        total_train_loss = 0
        total_train_correct = [0] * num_tasks
        total_train_samples = [0] * num_tasks
        num_train_batches = min(len(dataloader) for dataloader in train_dataloaders)
        epoch_train_task_losses = [0] * num_tasks

        for batch_idx, batches in enumerate(zip(*train_dataloaders)):
            optimizer.zero_grad()
            task_losses = []
            weights = model.get_weights()

            for task_id, (X_batch, y_batch, mask, _) in enumerate(batches):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask = mask.to(device)

                output = model(X_batch, mask, task_id)
                loss = criterion(output, y_batch)
                task_losses.append(loss)

                # Accumulate the loss for each task
                epoch_train_task_losses[task_id] += loss.item()

                pred = output.argmax(dim=1)
                total_train_correct[task_id] += (pred == y_batch).sum().item()
                total_train_samples[task_id] += y_batch.size(0)

            weighted_loss = sum(w * loss for w, loss in zip(weights, task_losses))
            weighted_loss.backward()
            optimizer.step()

            total_train_loss += weighted_loss.item()

        # Calculate the average training loss and accuracy
        avg_train_loss = total_train_loss / num_train_batches
        train_accuracies = [correct / samples for correct, samples in zip(total_train_correct, total_train_samples)]
        avg_train_accuracy = sum(train_accuracies) / num_tasks

        # Compute the average training loss for each task
        epoch_train_task_losses = [loss / num_train_batches for loss in epoch_train_task_losses]
        train_task_losses.append(epoch_train_task_losses)

        total_train_losses.append(avg_train_loss)
        total_train_accuracies.append(avg_train_accuracy)

        model.eval()
        total_val_loss = 0
        total_val_correct = [0] * num_tasks
        total_val_samples = [0] * num_tasks
        num_val_batches = min(len(dataloader) for dataloader in val_dataloaders)
        epoch_val_task_losses = [0] * num_tasks

        with torch.no_grad():
            for batches in zip(*val_dataloaders):
                task_losses = []
                weights = model.get_weights()

                for task_id, (X_batch, y_batch, mask, _) in enumerate(batches):
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    mask = mask.to(device)

                    output = model(X_batch, mask, task_id)
                    loss = criterion(output, y_batch)
                    task_losses.append(loss)

                    # Accumulate the loss for each task
                    epoch_val_task_losses[task_id] += loss.item()

                    pred = output.argmax(dim=1)
                    total_val_correct[task_id] += (pred == y_batch).sum().item()
                    total_val_samples[task_id] += y_batch.size(0)

                weighted_loss = sum(w * loss for w, loss in zip(weights, task_losses))
                total_val_loss += weighted_loss.item()

        # Calculate the average validation loss and accuracy
        avg_val_loss = total_val_loss / num_val_batches
        val_accuracies = [correct / samples for correct, samples in zip(total_val_correct, total_val_samples)]
        avg_val_accuracy = sum(val_accuracies) / num_tasks

        # Compute the average validation loss for each task
        epoch_val_task_losses = [loss / num_val_batches for loss in epoch_val_task_losses]
        val_task_losses.append(epoch_val_task_losses)

        total_val_losses.append(avg_val_loss)
        total_val_accuracies.append(avg_val_accuracy)

        # update lr if val_loss increase beyond patience
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']

        if best_acc < avg_val_accuracy:
            best_acc = avg_val_accuracy
            torch.save(model.state_dict(), os.path.join(savepath, 'best_model.pth'))

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Output the training and validation results
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"lr: {lr:.6f},"
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Task Losses: {[f'Task {i}: {loss:.4f}' for i, loss in enumerate(epoch_train_task_losses)]}, "
              f"Train Avg Accuracy: {avg_train_accuracy:.4f}, "
              f"Train Accuracies: {[f'Task {i}: {acc:.4f}' for i, acc in enumerate(train_accuracies)]}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Task Losses: {[f'Task {i}: {loss:.4f}' for i, loss in enumerate(epoch_val_task_losses)]}, "
              f"Val Avg Accuracy: {avg_val_accuracy:.4f}, "
              f"Val Accuracies: {[f'Task {i}: {acc:.4f}' for i, acc in enumerate(val_accuracies)]}, "
              f"Weights: {[w.item() for w in weights]}")
    return {
        'total_train_losses': total_train_losses,
        'total_val_losses': total_val_losses,
        'total_train_accuracies': total_train_accuracies,
        'total_val_accuracies': total_val_accuracies,
        'train_task_losses': train_task_losses,
        'val_task_losses': val_task_losses
    }


def test_model(model, test_dataloaders, task_id):
    """
    Perform inference on the test set and generate predictions.

    :param model: The trained multitask Transformer model.
    :param test_dataloaders: Single DataLoader for a specific task or a list of DataLoaders for multiple tasks.
    :param task_id: Optional integer specifying the task ID for single-task inference; if None, performs multitask inference.

    :return:
        - If task_id is specified: predictions and their corresponding probabilities for the specified task.
        - If task_id is None: containing predictions for each task.
    """

    model.eval()

    # Determine if single-task or multi-task mode
    is_single_task = isinstance(test_dataloaders, DataLoader)
    if is_single_task and task_id is None:
        raise ValueError("task_id must be specified when test_dataloaders is a single DataLoader")

    # Single-task inference
    if is_single_task:
        predictions = []
        probabilities = []
        with torch.no_grad():
            for X_batch, _, mask, _ in test_dataloaders:
                X_batch = X_batch.to(device)
                mask = mask.to(device)
                output = model(X_batch, mask, task_id)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)

                predictions.extend(pred.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())

        return predictions, probabilities

    # Multi-task inference
    else:
        num_tasks = len(test_dataloaders)
        predictions = [[] for _ in range(num_tasks)]
        with torch.no_grad():
            for batches in zip(*test_dataloaders):
                for t_id, (X_batch, _, mask, _) in enumerate(batches):
                    X_batch = X_batch.to(device)
                    mask = mask.to(device)
                    output = model(X_batch, mask, t_id)
                    pred = output.argmax(dim=1)
                    predictions[t_id].extend(pred.cpu().tolist())

        return predictions
