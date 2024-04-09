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
from model import FusionModel, LinearClassifier
from loss import info_rank_loss

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
    temperature = kwargs['temperature']

    # Create save location using slune and tensorboard writer
    formatted_args = dict_to_ls(**kwargs)
    saver = slune.get_csv_saver(formatted_args, root_dir='results')
    path = os.path.dirname(saver.get_current_path())
    writer = SummaryWriter(path)    

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using {device} device ---", flush=True)
    print(f"--- Using following config {kwargs} ---", flush=True)
            
    # Generate / Load in the data
    if benchmark == "written_spoken_digits":
        train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    else:
        raise ValueError("Invalid benchmark")
    
    if model == "FusionModel":
        model = FusionModel()
    else:
        raise ValueError("Invalid model")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define the loss function based on the estimator
    if est == "info_rank":
        loss_fun = info_rank_loss
    else:
        raise ValueError("Invalid est(imator)")
    
    # Set-up for training
    losses = []
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping

    # Train the model
    cum_b = -1
    for epoch in range(num_epochs):
        for b, (image_batch, audio_batch, label_batch) in enumerate(train_loader):
            cum_b += 1
            image_batch, audio_batch = image_batch.to(device), audio_batch.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = loss_fun(model, image_batch, audio_batch, temperature, device)
            losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), cum_b)
            saver.log({'train_loss': loss.item()})

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


        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping", flush=True)
                break

    # Now that we have trained the model, we evaluate it by training a linear classifier on top of the frozen representations
    # Define the linear classifier
    linear_classifier = LinearClassifier(64, 10).to(device)
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
            if est == "info_rank":
                rep = model(image_batch, audio_batch)
            else:
                raise ValueError("Invalid est(imator)")
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
            if est == "info_rank":
                rep = model(image_test, audio_test)
            else:
                raise ValueError("Invalid est(imator)")
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

    saver.save_collated()
    return losses, model

# Main function used to execute a training example and visualize results, use --default_exp_name to specify which example to run
# Use: python train.py --default_exp_name=bi_nce
    # To run the bi-NCE example
# Use: python train.py --default_exp_name=info_nce
    # To run the info-NCE example
if __name__ == "__main__":
    #  Parse input from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_exp_name', type=str, help='Name of the example experiment you want to run, currenly only info_rank', default="info_rank")
    args = parser.parse_args()
    config = {
        'benchmark': 'written_spoken_digits',
        'model': 'FusionModel',
        'learning_rate': 1e-7,
        'num_epochs': 200,
        'batch_size': 512,
        'est': args.default_exp_name,
        'patience': 5,
        'temperature': 0.1,
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
