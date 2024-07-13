from data import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import dict_to_ls
import slune
import os
from model import FusionModel, LinearClassifier, StrictFusionModel, ShallowStrictFusionModel
from loss import SimCLR_loss, info_critic, info_critic_plus, info_rank, info_rank_plus, prob_loss, decomposed_loss, get_train_accuracy

def train(**kwargs):
    """
    Train the model using the specified estimator.

    Args:
        benchmark: Literal['written_spoken_digits'] (dataset to use)
        model: Literal['FusionModel'] (model to train)
        learning_rate: float (learning rate for optimizer)
        num_epochs: int (number of epochs to train for)
        batch_size: int (batch size for training, if None, use full dataset)
        est: Literal['info_rank'] (estimator to use)
        patience: int (number of epochs to wait before early stopping)
        temperature: float (temperature for the estimator)

    Returns:
        losses: list
        accuracies: list 
        model: nn.Module

    """
    # Unpack the config
    benchmark = kwargs['benchmark']
    model = kwargs['model']
    learning_rate = kwargs['learning_rate']
    num_epochs = kwargs['num_epochs']
    batch_size = kwargs['batch_size']
    est = kwargs['est']
    patience = kwargs['patience']
    if patience < 0:
        patience = None
    temperature = kwargs['temperature']

    # Create save location using slune and tensorboard writer
    saver = slune.get_csv_saver(kwargs, root_dir='results')
    path = os.path.dirname(saver.getset_current_path())
    writer = SummaryWriter(path)    

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using {device} device ---", flush=True)
    print(f"--- Using following config {kwargs} ---", flush=True)
            
    # Generate / Load in the data
    if benchmark == "written_spoken_digits":
        train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    else:
        raise ValueError("Invalid benchmark: {}".format(benchmark))
    
    if model == "FusionModel":
        model = FusionModel()
    elif model == "ImageOnly":
        model = FusionModel(single_modality="image")
    elif model == "AudioOnly":
        model = FusionModel(single_modality="audio")
    elif model == "StrictFusionModel":
        model = StrictFusionModel()
    elif model == "StrictImageOnly":
        model = StrictFusionModel(single_modality="image")
    elif model == "StrictAudioOnly":   
        model = StrictFusionModel(single_modality="audio")
    elif model == "ShallowStrictFusionModel":
        model = ShallowStrictFusionModel()
    elif model == "ShallowStrictImageOnly":
        model = ShallowStrictFusionModel(single_modality="image")
    elif model == "ShallowStrictAudioOnly":
        model = ShallowStrictFusionModel(single_modality="audio")
    else:
        raise ValueError("Invalid model")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define the loss function based on the estimator
    if est == "info_critic":
        loss_fun = info_critic
    elif est == "info_critic_plus":
        loss_fun = info_critic_plus
    elif est == "info_rank":
        loss_fun = info_rank
    elif est == "info_rank_plus":
        loss_fun = info_rank_plus
    elif est == "prob_loss":
        loss_fun = prob_loss
    elif est == "decomposed_loss":
        loss_fun = decomposed_loss
    elif est == "SimCLR":
        loss_fun = SimCLR_loss
    else:
        raise ValueError("Invalid est(imator)")
    
    # Set-up for training
    losses = []
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping

    # Train the model
    cum_b = -1
    epoch_losses = []
    epoch_train_accuracies = []
    batch_stop = len(train_loader)
    if  1 > num_epochs > 0:
        batch_stop = num_epochs * len(train_loader)
        num_epochs = 1
    num_epochs = int(num_epochs)
    for e, epoch in enumerate(range(num_epochs)):
        for b, (image_batch, audio_batch, label_batch) in enumerate(train_loader):
            if b >= batch_stop:
                break
            # Increment the batch counter
            cum_b += 1
            image_batch, audio_batch = image_batch.to(device), audio_batch.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = loss_fun(model, image_batch, audio_batch, temperature, device)
            losses.append(loss.item())
            epoch_losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), cum_b)
            saver.log({'train_loss': loss.item()})

            # Calculate and log training accuracy
            train_acc = get_train_accuracy(model, image_batch, audio_batch, est, device)
            epoch_train_accuracies.append(train_acc.item())
            writer.add_scalar('Accuracy/train', train_acc.item(), cum_b)
            saver.log({'train_acc': train_acc.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return losses, model
        
        if (epoch + 1) % 25 == 0:
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

        # Avg. epoch loss
        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('Loss/epoch_avg_train', avg_loss, e)
        saver.log({'train_loss_epoch_avg': avg_loss})
        epoch_losses = []

        # Avg. epoch train accuracy
        avg_train_acc = np.mean(epoch_train_accuracies)
        writer.add_scalar('Accuracy/epoch_avg_train', avg_train_acc, e)
        saver.log({'train_acc_epoch_avg': avg_train_acc})
        epoch_train_accuracies = []

        # Early stopping
        if patience is None:
            pass
        elif avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping at epoch:"+str(epoch+1), flush=True)
                break

    # Now that we have trained the model, we evaluate it by training a linear classifier on top of the frozen representations
    # Define the linear classifier
    # Find output size of network
    linear_classifier = LinearClassifier(model.output_dim, 10).to(device)
    learning_rate = 0.1 * batch_size / 256
    optimizer = optim.SGD(linear_classifier.parameters(), lr=learning_rate)
    for param in model.parameters():
        param.requires_grad = False

    # Train the linear classifier
    cum_b = -1
    for epoch in range(50):
        for b, (image_batch, audio_batch, label_batch) in enumerate(train_loader):
            cum_b += 1
            image_batch, audio_batch, label_batch = image_batch.to(device), audio_batch.to(device), label_batch.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            with torch.no_grad():
                rep = model(image_batch, audio_batch)
            logits = linear_classifier(rep)
            loss = nn.CrossEntropyLoss()(logits, label_batch)
            writer.add_scalar('Eval/train_loss', loss.item(), cum_b)
            saver.log({'eval_train_loss': loss.item()})
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return losses, model
            
            cum_b += 1
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

    # Test the linear classifier
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for image_test, audio_test, label_test in test_loader:
            image_test, audio_test, label_test = image_test.to(device), audio_test.to(device), label_test.to(device)
            rep = model(image_test, audio_test)
            logits = linear_classifier(rep)
            loss = nn.CrossEntropyLoss()(logits, label_test)
            total_loss += loss.item() * image_test.size(0)  # Multiply by batch size
            # Compute the accuracy
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == label_test).sum().item()
            total_samples += image_test.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    writer.add_scalar('Eval/test_loss', avg_loss, cum_b)
    saver.log({'eval_test_loss': avg_loss})
    writer.add_scalar('Eval/test_accuracy', accuracy, cum_b)
    saver.log({'eval_test_accuracy': accuracy})
    print(f'Accuracy of the network on the {total_samples} test images: {accuracy:.4f}', flush=True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for image_train, audio_train, label_train in train_loader:
            image_train, audio_train, label_train = image_train.to(device), audio_train.to(device), label_train.to(device)
            rep = model(image_train, audio_train)
            logits = linear_classifier(rep)
            loss = nn.CrossEntropyLoss()(logits, label_train)
            writer.add_scalar('Eval/train_loss', loss.item(), cum_b)
            saver.log({'eval_train_loss': loss.item()})
            # Compute the accuracy
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == label_train).sum().item()
            total_samples += image_train.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    writer.add_scalar('Eval/train_loss', avg_loss, cum_b)
    saver.log({'eval_train_loss': avg_loss})
    writer.add_scalar('Eval/train_accuracy', accuracy, cum_b)
    saver.log({'eval_train_accuracy': accuracy})
    print(f'Accuracy of the network on the {total_samples} train images: {accuracy:.4f}', flush=True)

    saver.save_collated()
    return losses, model

# Main function used to execute a training example and visualize results, use --est to specify which estimator to use in training run

if __name__ == "__main__":
    #  Parse input from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--est', type=str, help='Name of the estimator you want to use, currenly only info_rank or SimCLR', default="info_critic")
    args = parser.parse_args()
    config = {
        'benchmark': 'written_spoken_digits',
        'model': 'FusionModel',
        'learning_rate': 1e-4,
        'num_epochs': 1,
        'batch_size': 128,
        'est': args.est,
        'patience': -1,
        'temperature': 1,
    }
    # Train the model
    losses, model = train(**config)

    # Create save location using slune
    formatted_args = dict_to_ls(**config)
    saver = slune.get_csv_saver(formatted_args, root_dir='results')
    path = os.path.dirname(saver.get_current_path())
    # Save the model to the save location using torch.save
    torch.save(model, os.path.join(path, "model.pt"))

    # Plot the loss
    plt.figure()
    plt.plot(np.arange(len(losses)), np.array(losses))
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()
