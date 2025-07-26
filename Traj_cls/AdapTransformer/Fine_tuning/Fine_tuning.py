# -*- coding: utf-8 -*-
# @Time    : 2025/4/3 16:47
# @Author  : Dongliang


import os
import pickle
from datetime import datetime

import scipy
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from AAISPT.Traj_cls.AdapTransformer.dataset import collate_fn
from AAISPT.Traj_cls.AdapTransformer.model import MultiTaskTransformer
from AAISPT.pytorch_tools import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, scheduler, criterion, device, savepath,
                early_stopping, ):
    """
    Train and validate a model, computing training and validation losses and accuracies.

    :param model: The model to train (e.g., MultiTaskTransformer).
    :param train_dataloader: DataLoader for training data.
    :param val_dataloader: DataLoader for validation data.
    :param num_epochs: Number of training epochs.
    :param optimizer: Optimizer for model parameters.
    :param scheduler: Learning rate scheduler.
    :param criterion: Loss function (e.g., CrossEntropyLoss).
    :param device: Device to run the model on (e.g., CPU or GPU).
    :param savepath: Directory path to save the best model.
    :param early_stopping: Early stopping mechanism callback.
    :return: A dictionary containing lists of training and validation.
            - 'train_losses': List of average training losses per epoch.
            - 'val_losses': List of average validation losses per epoch.
            - 'train_acces': List of training accuracies per epoch.
            - 'val_acces': List of validation accuracies per epoch.
    """
    model.to(device)
    train_losses, val_losses = [], []
    train_acces, val_acces = [], []
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total_nums = 0, 0, 0
        num_train_batches = len(train_dataloader)

        for X_batch, y_batch, mask, _ in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = model(X_batch, mask, task_id=0)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total_nums += y_batch.size(0)

        avg_train_loss = train_loss / num_train_batches
        train_accuracy = correct / total_nums
        train_losses.append(avg_train_loss)
        train_acces.append(train_accuracy)

        model.eval()
        val_loss, correct, total_nums = 0, 0, 0
        num_val_batches = len(val_dataloader)

        with torch.no_grad():
            for X_batch, y_batch, mask, _ in val_dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask = mask.to(device)

                output = model(X_batch, mask, task_id=0)
                loss = criterion(output, y_batch)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total_nums += y_batch.size(0)

        avg_val_loss = val_loss / num_val_batches
        val_accuracy = correct / total_nums
        val_losses.append(avg_val_loss)
        val_acces.append(val_accuracy)

        # update lr if val_loss increase beyond patience
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        if best_acc < val_accuracy:
            best_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(savepath, 'best_model.pth'))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Learning Rate: {lr:.6f}, '
              f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} ')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acces': train_acces,
        'val_acces': val_acces
    }


def fine_tune():
    path = r'/root/autodl-tmp/4D_Finetune'
    savepath = os.path.join(path, 'AdapTransformer')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    if '2D' in path:
        feature_dims = [32]
        num_classes_list = [3]
    elif '4D' in path:
        feature_dims = [46]
        num_classes_list = [3]
    elif '5D' in path:
        feature_dims = [46]
        num_classes_list = [4]

    # Load the preprocessed Dataset
    train_dataset = torch.load(os.path.join(savepath, 'train_dataset.pt'))
    val_dataset = torch.load(os.path.join(savepath, 'val_dataset.pt'))

    # Set hyperparameters
    batch_size = 256
    init_lr = 0.001
    weight_decay = 0.0001
    epochs = 100

    # Set network parameters (consistent with pre-trained model)
    d_model = 64
    num_layers = 2
    num_heads = 4
    hidden_dim = 128

    # Create the DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load the pre-trained model
    pretrained_model_path = r'/root/autodl-tmp/4D_Finetune/AdapTransformer/AdapTransformer_model.pth'
    pretrained_state_dict = torch.load(pretrained_model_path)

    # Initialize the new model
    model = MultiTaskTransformer(
        feature_dims=feature_dims,
        num_classes_list=num_classes_list,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim
    )

    # Load pre-training weights
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.startswith('projs.1'):
            # Reuse the input projection layer from task_id=1
            new_k = k.replace('projs.1', 'projs.0')  # Map to a new single-task model
            pretrained_dict[new_k] = v
        elif k.startswith('fcs.1'):  # Reuse the output classification layer from task_id=1
            new_k = k.replace('fcs.1', 'fcs.0')
            continue
        elif not k.startswith('projs') and not k.startswith('fcs') and not k.startswith('logits'):
            pretrained_dict[k] = v
    # pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
    #                    k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Fine-tuned
    for name, param in model.named_parameters():
        param.requires_grad = True

        # Freeze the Transformer encoder, only fine-tune the input layer, output layer, and task weights
        # if 'projs' not in name and 'fcs' not in name:
        #     param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr,
                                 weight_decay=weight_decay)  # Optimize only parameters that are not frozen.
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, eps=1e-06)

    # Initialize the early_stopping object
    patience = 10
    early_stopping = EarlyStopping(patience, verbose=True)

    train_losses, val_losses, train_acces, val_acces = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        savepath=savepath,
        early_stopping=early_stopping,

    )

    # Save the fine-tuned model
    torch.save(model.state_dict(), os.path.join(savepath, 'finetuned_model.pth'))

    history = {
        'train_losses': train_losses,
        'train_acces': train_acces,
        'val_losses': val_losses,
        'val_acces': val_acces,
    }

    with open(os.path.join(savepath, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print('======================== Visualization =======================')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(train_acces)
    plt.plot(val_acces)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(savepath, 'Loss_Acc.png'), dpi=600)
    plt.close()

    params = {
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': init_lr,
        'weight_decay': weight_decay,
        'savepath': savepath
    }

    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = os.path.join(savepath, f'params_{current_time}.txt')
    with open(file_name, 'w') as f:
        f.write('Train parameters:\n')
        for key, value in params.items():
            f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    # fine_tune()

    path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Roll_Feature\AdapTransformer'
    model_path = r'D:\TrajSeg-Cls\TrajSEG-CLS_V3\CLS\YanYu_NEW\Roll_Feature\AdapTransformer\全参微调\Pre-trained model from All dimensional'
    savepath = model_path
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    batch_size = 2048

    # Load the preprocessed Dataset
    test_dataset = torch.load(os.path.join(path, 'test_dataset.pt'))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Network parameters (fixed)
    d_model = 64
    num_layers = 2
    num_heads = 4
    hidden_dim = 128

    if 'QiPan' in path:
        feature_dims = [32]
        num_classes_list = [3]
    elif 'YanYu' in path:
        feature_dims = [46]
        num_classes_list = [3]
    elif 'Endocytosis' in path:
        feature_dims = [46]
        num_classes_list = [4]

    # Initialize the model
    model = MultiTaskTransformer(
        feature_dims=feature_dims,
        num_classes_list=num_classes_list,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim
    )

    model.load_state_dict(torch.load(os.path.join(model_path, 'finetuned_model.pth')))
    model.to(device)

    model.eval()
    predictions = []
    probabilities = []
    test_y = []
    with torch.no_grad():
        for X_batch, y_batch, mask, _ in test_dataloader:
            X_batch = X_batch.to(device)
            mask = mask.to(device)

            output = model(X_batch, mask, task_id=0)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            predictions.extend(pred.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
            test_y.extend(y_batch)

    scipy.io.savemat(os.path.join(savepath, 'cls_pre.mat'),
                     {'clspre': predictions,
                      'clsgt': test_y,
                      'probs': probabilities
                      }
                     )
