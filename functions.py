import numpy as np
import os
import time
import warnings
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.rcParams['figure.figsize'] = [15, 7]

def accuracy(predictions, labels):
    #round predictions to the closest integer
    probs = torch.softmax(predictions, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == labels).float()
    acc = corrects.sum() / len(corrects)
    return acc

def train_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc  = 0
    model.train()
    for batch in iterator:

        optimizer.zero_grad()

        img, label, metadata = batch
        img = img.to(device)
        label = label.to(device)
        metadata = metadata.to(device)

        predictions = model(img, metadata)

        loss = criterion(predictions, label)
        acc  = accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc  += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)

def eval_epoch(model, iterator, criterion, device):

    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            img, label, metadata = batch
            img = img.to(device)
            label = label.to(device)
            metadata = metadata.to(device)

            #convert to 1d tensor
            predictions = model(img, metadata)

            #compute loss and accuracy
            loss = criterion(predictions, label)
            acc = accuracy(predictions, label)

            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def testing(model, iterator, batch_size, device):

    preds = []
    lbls  = []
    predictions = torch.Tensor(len(iterator)*batch_size).to(device)
    labels      = torch.Tensor(len(iterator)*batch_size).to(device)

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:

            img, label, metadata = batch
            img = img.to(device)
            label = label.to(device)
            metadata = metadata.to(device)

            #convert to 1d tensor
            scores = model(img, metadata)
            probs = torch.softmax(scores, dim=1)
            winners = probs.argmax(dim=1)
            preds.append(winners)
            lbls.append(label)

    torch.cat(preds, out=predictions)
    torch.cat(lbls, out=labels)

    return predictions.cpu().numpy(), labels.cpu().numpy()

def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs,
          saved_models_folder, salve_scores_folder, save_path, printfreq=10):

    train_loss_hist = []
    train_acc_hist  = []
    val_loss_hist   = []
    val_acc_hist    = []

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    t0 = time.time()
    print(' Epoch    Train Loss    Val Loss    Train Acc    Val Acc    Best    Time [min]')
    print('-'*77)

    for epoch in range(num_epochs):
        t1 = time.time()
        st = '        '

        # Training metrics
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # Validation metrics
        val_loss, val_acc     = eval_epoch(model, val_dataloader, criterion, device)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        # Update best model
        if val_loss < best_val_loss:
            st = '     ***'
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(saved_models_folder, 'best_' + save_path))

        if (epoch + 1) % printfreq == 0:
            t2 = (time.time() - t1)/60
            s = f'{epoch+1:6}{train_loss:12.4f}{val_loss:13.4f}{train_acc:13.4f}{val_acc:12.4f}{st}{t2:10.1f}'
            print(s)

        # Save current model and training information
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(saved_models_folder, save_path))

        # Save the loss and accuracy values
        np.save(os.path.join(saved_scores_folder, f'{save_path}_train_loss'), train_loss_hist)
        np.save(os.path.join(saved_scores_folder, f'{save_path}_train_acc'), train_acc_hist)
        np.save(os.path.join(saved_scores_folder, f'{save_path}_val_loss'), val_loss_hist)
        np.save(os.path.join(saved_scores_folder, f'{save_path}_val_acc'), val_acc_hist)

    tfinal = (time.time() - t0)/60
    print('-'*77)
    print(f'Total time [min] for {num_epochs} Epochs: {tfinal:.1f}')
    return

def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def metrics(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    Accuracy = np.sum(np.diag(cm))/np.sum(cm)
    Pr = np.mean(np.diag(cm) / np.sum(cm, axis = 0))
    Re = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
    F1 = 2*(Pr*Re)/(Pr + Re)
    return Accuracy, Pr, Re, F1, cm


def display_train_metrics(save_path, num_epochs):
    epochs = range(1, num_epochs + 1)

    # Loss
    train_loss_hist = np.load(os.path.join(saved_scores_folder, f'{save_path}_train_loss.npy'))
    val_loss_hist = np.load(os.path.join(saved_scores_folder, f'{save_path}_val_loss.npy'))

    # Acc
    train_acc_hist = np.load(os.path.join(saved_scores_folder, f'{save_path}_train_acc.npy'))
    val_acc_hist = np.load(os.path.join(saved_scores_folder, f'{save_path}_val_acc.npy'))

    plt.plot(epochs, train_loss_hist, marker='o', label='Training')
    plt.plot(epochs, val_loss_hist, marker='o', label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, train_acc_hist, marker='o', label='Training')
    plt.plot(epochs, val_acc_hist, marker='o', label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
