from data_digits import digits_get_data_loaders
from data_nyu_v2 import nyu_v2_get_data_loaders
from data_mosi_mosei import mosi_get_data_loaders, mosei_get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import MyDataParallel, log_memory, get_batch_labels, get_torchmetrics
import slune
import os
from model import FusionModel, LinearClassifier, ImageModel, AudioModel, ResNet101, SegClassifier, ResNet50, MosiFusion, MoseiFusion
from loss import SimCLR_loss, info_critic, info_critic_plus, prob_loss, decomposed_loss, get_train_accuracy, augmenter

def supervised_train(model, optimizer, train_loader, device, writer, saver, num_epochs, patience, modality='image+audio'):
    # Create classifier
    linear_classifier = LinearClassifier(model.output_dim, 10).to(device)
    # Set-up for training
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping
    cum_b = -1

    losses = []
    epoch_losses = []
    epoch_train_accuracies = []

    batch_stop = len(train_loader)
    if  1 > num_epochs > 0:
        batch_stop = num_epochs * len(train_loader)
        num_epochs = 1
    num_epochs = int(num_epochs)

    # Train the model
    for e, epoch in enumerate(range(num_epochs)):
        for b, batch in enumerate(train_loader):
            if b >= batch_stop:
                break
            # Increment the batch counter
            cum_b += 1
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            if modality == 'image+audio':
                image, audio, labels = batch
                batch = (image.to(device), audio.to(device))
            elif modality == 'image':
                image, _, labels = batch
                batch = image.to(device)
            elif modality == 'audio':
                _, audio, labels = batch
                batch = audio.to(device)
            else:
                raise ValueError("Invalid aug")
            rep = model(batch)
            logits = linear_classifier(rep)
            labels = labels.to(device)
            loss = F.cross_entropy(logits, labels)
            losses.append(loss.item())
            epoch_losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), cum_b)
            saver.log({'train_loss': loss.item()})
            
            # Calculate and log training accuracy
            train_acc = (logits.argmax(dim=1) == labels).float().mean()
            epoch_train_accuracies.append(train_acc.item())
            writer.add_scalar('Accuracy/train', train_acc.item(), cum_b)
            saver.log({'train_acc': train_acc.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return losses, model, linear_classifier
            
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
    return losses, model, linear_classifier
    
def unsupervised_train(model, optimizer, loss_fun, train_loader, est, temperature, device, writer, saver, num_epochs, patience, modality='image+audio'):
    # Set-up for training
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping

    # Train the model
    cum_b = -1
    batch_stop = len(train_loader)
    if  1 > num_epochs > 0:
        batch_stop = num_epochs * len(train_loader)
        num_epochs = 1
    num_epochs = int(num_epochs)
    scaler = torch.amp.GradScaler()
    for e, epoch in enumerate(range(num_epochs)):
        epoch_losses = []
        epoch_train_accs = []
        for b, batch in enumerate(train_loader):
            if b >= batch_stop:
                break
            # Increment the batch counter
            cum_b += 1
            # Augment the batch
            batch1, batch2 = augmenter(batch, modality, device)
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                loss = loss_fun(model, batch1, batch2, temperature, device)
            # Backward pass and optimization
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            # Log memory usage 
            log_memory(writer, cum_b)

            # Log the loss
            epoch_losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), cum_b)
            saver.log({'train_loss': loss.item()})

            # Calculate and log training accuracy
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                with torch.no_grad():
                    accs = get_train_accuracy(model, batch1, batch2, est, device)
            epoch_train_accs.append(accs)
            writer.add_scalar('Accuracy/train', accs['accuracy'].item(), cum_b)
            saver.log({'train_acc': accs['accuracy'].item()})

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return model
        
        if epoch % 10 == 0:
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

        # Avg. epoch loss
        avg_loss = np.mean(epoch_losses)
        writer.add_scalar('Loss/epoch_avg_train', avg_loss, e)
        saver.log({'train_loss_epoch_avg': avg_loss})

        for key in epoch_train_accs[0].keys():
            avg_train_acc = np.mean([x[key].item() for x in epoch_train_accs])
            writer.add_scalar('Accuracy/epoch_avg_'+key, avg_train_acc, e)
            saver.log({'train_acc_epoch_avg_'+key: avg_train_acc})

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
    return model
            
def eval_train(model, optimizer, train_loader, device, writer, saver, batch_size, modality='image+audio', classifier_type='linear_10'):
    # Define the linear classifier
    # Find output size of network
    if 'linear' in classifier_type:
        _, num_classes = classifier_type.split('_')
        num_classes = int(num_classes)
        classifier = LinearClassifier(model.output_dim, num_classes)
        classifier = MyDataParallel(classifier)
        classifier = classifier.to(device)
        criterion = nn.CrossEntropyLoss()
    elif 'seg' in classifier_type:
        _, num_classes = classifier_type.split('_')
        num_classes = int(num_classes)
        classifier = SegClassifier(num_classes)
        classifier = MyDataParallel(classifier)
        classifier = classifier.to(device)
        criterion = nn.CrossEntropyLoss()
    elif classifier_type == 'regr':
        classifier = LinearClassifier(model.output_dim, 1)
        classifier = MyDataParallel(classifier)
        classifier = classifier.to(device)
        criterion = nn.MSELoss()
    learning_rate = 0.1 * batch_size / 256
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    for param in model.parameters():
        param.requires_grad = False

    # Train the linear classifier
    cum_b = -1
    num_epochs = 50
    patience = 50 # For early stopping
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping
    scaler = torch.amp.GradScaler()
    for epoch in range(num_epochs):
        epoch_losses = []
        for b, batch in enumerate(train_loader):
            cum_b += 1
            batch, labels = get_batch_labels(modality, batch, device)
            with torch.no_grad():
                rep = model(batch)
    
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                optimizer.zero_grad()
                logits = classifier(rep)
                loss = criterion(logits, labels)

            # Backward pass and optimization
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # We zero gradients to save memory

            epoch_losses.append(loss.item())
            writer.add_scalar('Eval/train_loss', loss.item(), cum_b)
            saver.log({'eval_train_loss': loss.item()})

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return model, classifier
            
            cum_b += 1
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

        # Early stopping
        avg_loss = np.mean(epoch_losses)
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
    return model, classifier

def test(model, classifier, data_loader, device, writer, saver, name='test', modality='image+audio'):
    # Test the classifier
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    num_classes = classifier.num_classes
    metrics = get_torchmetrics(modality, num_classes, device)
    if modality == 'image_ft+audio_ft+text':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in data_loader:
            batch, labels = get_batch_labels(modality, batch, device)
            rep = model(batch)
            logits = classifier(rep)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_batches += 1
            total_samples += len(labels)
            # Compute metrics 
            _, predicted = torch.max(logits.data, 1)
            if len(predicted.shape) ==1:
                predicted = predicted.unsqueeze(1)
            for metric in metrics.values():
                metric(predicted, labels)

    avg_loss = total_loss / total_batches
    writer.add_scalar('Eval/'+name+'_loss', avg_loss, 0)
    saver.log({'eval_'+name+'_loss': avg_loss})
    
    for key, metric in metrics.items():
        metric = metric.compute()
        writer.add_scalar('Eval/'+name+'_'+key, metric, 0)
        saver.log({'eval_'+name+'_'+key: metric})
        print(f'{key} of the network on the {total_samples} images: {metric:.4f}', flush=True)
        metrics[key].reset()

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
        output_dim: int (output dimension of the model)

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
    output_dim = kwargs['output_dim']

    # Create save location using slune and tensorboard writer
    saver = slune.get_csv_saver(kwargs, root_dir='results')
    path, _ = os.path.splitext(saver.getset_current_path())
    writer = SummaryWriter(path)    

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using {device} device ---", flush=True)
    print(f"--- Using following config {kwargs} ---", flush=True)
            
    # Generate / Load in the data
    if benchmark == "written_spoken_digits":
        train_loader, test_loader = digits_get_data_loaders(batch_size=batch_size)
        eval_train_loader = train_loader
    elif benchmark == "nyu_v2_13":
        train_loader, _, eval_train_loader, test_loader = nyu_v2_get_data_loaders(batch_size=batch_size, num_classes=13)
    elif benchmark == "nyu_v2_40":
        train_loader, _, eval_train_loader, test_loader = nyu_v2_get_data_loaders(batch_size=batch_size, num_classes=40)
    elif benchmark == "mosi":
        train_loader, _, test_loader = mosi_get_data_loaders(batch_size=batch_size)
        eval_train_loader = train_loader
    elif benchmark == "mosei":
        train_loader, _, test_loader = mosei_get_data_loaders(batch_size=batch_size)
        eval_train_loader = train_loader
    else:
        raise ValueError("Invalid benchmark: {}".format(benchmark))
    
    if model == "FusionModel":
        model = FusionModel(output_dim=output_dim)
        modality = 'image+audio'
        classifier_type = 'linear_10'
    elif model == "ImageOnly":
        model = ImageModel(output_dim=output_dim)
        modality = 'image'
        classifier_type = 'linear_10'
    elif model == "AudioOnly":
        model = AudioModel(output_dim=output_dim)
        modality = 'audio'
        classifier_type = 'linear_10'
    elif model == "ResNet101":
        model = ResNet101(output_dim=output_dim)
        model = MyDataParallel(model)
        modality = 'image+depth'
        classifier_type = 'seg_14' if benchmark == "nyu_v2_13" else 'seg_41'
    elif model == "ResNet50":
        model = ResNet50(output_dim=output_dim)
        model = MyDataParallel(model)
        modality = 'image+depth'
        classifier_type = 'seg_14' if benchmark == "nyu_v2_13" else 'seg_41'
    elif model == "MosiFusion":
        model = MosiFusion(output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
        classifier_type = 'regr'
    elif model == "MoseiFusion":
        model = MoseiFusion(output_dim=output_dim)
        modality = 'image_ft+audio_ft+text'
        classifier_type = 'regr'
    else:
        raise ValueError("Invalid model")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # If we want to do supervised training, otherwise continue on to unsupervised training
    if est == "supervised" and benchmark == "written_spoken_digits":
        losses, model, linear_classifier = supervised_train(model, optimizer, train_loader, device, writer, saver, num_epochs, patience, modality=modality)
        test(model, linear_classifier, test_loader, device, writer, saver, name='test', modality=modality)
        test(model, linear_classifier, train_loader, device, writer, saver, name='train', modality=modality)
        saver.save_collated()
        return model, linear_classifier
    elif est == "supervised":
        raise ValueError("Invalid estimator for benchmark")

    # Define the loss function based on the estimator
    if est == "info_critic":
        loss_fun = info_critic
    elif est == "info_critic_plus":
        loss_fun = info_critic_plus
    elif est == "prob_loss":
        loss_fun = prob_loss
    elif est == "decomposed_loss":
        loss_fun = decomposed_loss
    elif est == "SimCLR":
        loss_fun = SimCLR_loss
    else:
        raise ValueError("Invalid est(imator)")

    # Train the model
    model = unsupervised_train(model, optimizer, loss_fun, train_loader, est, temperature, device, writer, saver, num_epochs, patience, modality=modality)

    # Now that we have trained the model, we evaluate it by training a linear classifier on top of the frozen representations
    model, classifier = eval_train(model, optimizer, eval_train_loader, device, writer, saver, batch_size, modality=modality, classifier_type=classifier_type)

    # Evaluate the model on the test set
    test(model, classifier, test_loader, device, writer, saver, 'test', modality=modality)

    # Evaluate the model on the train set
    test(model, classifier, eval_train_loader, device, writer, saver, 'train', modality=modality)

    saver.save_collated()
    return model, classifier

# Main function used for testing the training function
if __name__ == "__main__":
    #  Parse input from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Name of the benchmark you want to use, default written_spoken_digits', default="written_spoken_digits")
    parser.add_argument('--est', type=str, help='Name of the estimator you want to use, currenly only info_rank or SimCLR', default="SimCLR")
    parser.add_argument('--model', type=str, help='Name of the model you want to use, currenly only FusionModel', default="FusionModel")
    args = parser.parse_args()
    config = {
        'benchmark': args.benchmark,
        'est': args.est,
        'model': args.model,
        'learning_rate': 1e-8, 
        'num_epochs': 0.0003,
        'batch_size': 16, #16
        'patience': 10,
        'temperature': 1,
        'output_dim': 128, #2028
    }
    # Train the model
    model = train(**config)

    # Create save location using slune
    saver = slune.get_csv_saver(config, root_dir='results')
    path = os.path.dirname(saver.getset_current_path())
    # Save the model to the save location using torch.save
    torch.save(model, os.path.join(path, "model.pt"))