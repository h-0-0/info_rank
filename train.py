from data_digits import digits_get_data_loaders, digits_get_data_loaders_weak_modality, digits_get_data_loaders_noisy_pairing
from data_nyu_v2 import nyu_v2_get_data_loaders
from data_mosi_mosei import mosi_get_data_loaders, mosei_get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import MyDataParallel, log_memory, get_batch_labels, get_torchmetrics, get_model, get_classifier_criterion
import slune
import os
from model import FusionModel, LinearClassifier, ImageModel, AudioModel, ResNet101, SegClassifier, ResNet50, MosiFusion, MoseiFusion, Regression, FullConvNet, FCN_SegHead
from loss import SimCLR_loss, info_critic, info_critic_plus, prob_loss, decomposed_loss, get_train_accuracy, augmenter
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_valid_loss(model, classifier, criterion, valid_loader, modality, device):
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in valid_loader:
            batch, labels = get_batch_labels(modality, batch, device)
            rep = model(batch)
            logits = classifier(rep)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches

def supervised_train(model, classifier, criterion, train_loader, device, writer, saver, num_epochs, patience, learning_rate, modality='image+audio', optimizer_type='SGD', grad_clip=None, scheduler=None, valid_loader=None):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    # Set-up for training
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping
    cum_b = -1

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
            torch.autograd.set_detect_anomaly(True) #TODO: remove
            batch, labels = get_batch_labels(modality, batch, device)
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                rep = model(batch)
                logits = classifier(rep)
                labels = labels.to(device)
                loss = criterion(logits, labels)
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
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return model, classifier
            
            if scheduler is not None:
                if valid_loader is None:
                    raise ValueError("Need a validation loader to use scheduler")
                val_loss = get_valid_loss(model, classifier, criterion, valid_loader, modality, device)
                scheduler.step(val_loss) 
            
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
    return model, classifier
    
def unsupervised_train(model, optimizer, loss_fun, train_loader, est, temperature, device, writer, saver, num_epochs, patience, modality='image+audio', grad_clip=None):
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
    # scaler = torch.amp.GradScaler() #TODO: remove
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
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # scaler.scale(loss).backward() #TODO: remove
            optimizer.step()
            # scaler.step(optimizer) #TODO: remove
            # scaler.update() #TODO: remove

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
            acc = accs['accuracy'].item() if torch.is_tensor(accs['accuracy']) else accs['accuracy']
            writer.add_scalar('Accuracy/train', acc, cum_b)
            saver.log({'train_acc': acc})

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
            if torch.is_tensor(epoch_train_accs[0][key]):
                avg_train_acc = np.mean([x[key].item() for x in epoch_train_accs])
            else:
                avg_train_acc = np.mean([x[key] for x in epoch_train_accs])
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
            
def eval_train(model, classifier, criterion, optimizer, train_loader, device, writer, saver, lr, num_epochs, patience, modality='image+audio', grad_clip=None):
    optimizer = optim.SGD(classifier.parameters(), lr=lr)
    for param in model.parameters():
        param.requires_grad = False

    # Train the linear classifier
    cum_b = -1
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping
    # scaler = torch.amp.GradScaler() #TODO: remove
    for e, epoch in enumerate(range(num_epochs)):
        epoch_losses = []
        for b, batch in enumerate(train_loader):
            cum_b += 1
            batch, labels = get_batch_labels(modality, batch, device)
            with torch.no_grad():
                rep = model(batch)
    
            with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                logits = classifier(rep)
                loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward() 
            # scaler.scale(loss).backward() #TODO: remove
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step() 
            # scaler.step(optimizer) #TODO: remove
            # scaler.update() #TODO: remove
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
        writer.add_scalar('Eval/epoch_avg_train_loss', avg_loss, e)
        saver.log({'eval_train_loss_epoch_avg': avg_loss})
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

def test(model, classifier, criterion, data_loader, device, writer, saver, name='test', modality='image+audio'):
    # Test the classifier
    model.eval()  # Set the model to evaluation mode
    classifier.eval()  # Set the classifier to evaluation mode
    total_loss = 0.0
    total_samples = 0
    total_batches = 0
    num_classes = classifier.num_classes
    metrics = get_torchmetrics(modality, num_classes, device)
    if modality == 'image_ft+audio_ft+text':
        # Regression task
        get_predicitons = lambda x: x
    else:
        # Classification task
        get_predicitons = lambda x: torch.max(x, 1)[1]
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
            predicted = get_predicitons(logits)
            if len(predicted.shape) !=1:
                predicted = predicted.squeeze(1)
            if len(labels.shape) !=1:
                labels = labels.squeeze(1)
            for metric in metrics.values():
                metric(predicted, labels)

    avg_loss = total_loss / total_batches
    writer.add_scalar('Eval/'+name+'_loss', avg_loss, 0)
    saver.log({'eval_'+name+'_loss': avg_loss})
    
    for key, metric in metrics.items():
        metric = metric.compute()
        if isinstance(metric, torch.Tensor):
            metric = metric.item()
        writer.add_scalar('Eval/'+name+'_'+key, metric, 0)
        saver.log({'eval_'+name+'_'+key: metric})
        print(f'{key} of the network on the {total_samples} images: {metric:.4f}', flush=True)
        metrics[key].reset()

def train(**kwargs):
    """
    Train the model using the specified estimator.
    TODO: update args and return
    Args:
        benchmark: Literal['written_spoken_digits', 'nyu_v2_13', 'nyu_v2_40', 'mosi', 'mosei'] (dataset to use)
        model: Literal['FusionModel'] (model to train)
        learning_rate: float (learning rate for optimizer)
        num_epochs: int (number of epochs to train for)
        batch_size: int (batch size for training, if None, use full dataset)
        est: Literal['info_rank'] (estimator to use)
        patience: int (number of epochs to wait before early stopping)
        temperature: float (temperature for the estimator)
        output_dim: int (output dimension of the model)
        optimizer: Literal['SGD', 'Adam'] (optimizer to use)

    Returns:
        losses: list
        accuracies: list 
        model: nn.Module

    """
    # Unpack the config
    benchmark = kwargs['benchmark']
    if benchmark == "written_spoken_digits_weak_image" or benchmark == "written_spoken_digits_weak_audio" or "written_spoken_digits_noisy_pairing":
        sigma = kwargs.get('sigma', 0.1)
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
    optimizer = kwargs['optimizer']
    eval_lr = kwargs.get('eval_lr', 0.1 * batch_size / 256)
    eval_num_epochs = kwargs.get('eval_num_epochs', 50)
    eval_patience = kwargs.get('eval_patience', None)
    grad_clip = kwargs.get('grad_clip', None)
    scheduler = kwargs.get('scheduler', None)

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
    elif benchmark == "written_spoken_digits_weak_image":
        train_loader, test_loader = digits_get_data_loaders_weak_modality(batch_size=batch_size, sigma=sigma, weaken_image=True)
        eval_train_loader = train_loader
    elif benchmark == "written_spoken_digits_weak_audio":
        train_loader, test_loader = digits_get_data_loaders_weak_modality(batch_size=batch_size, sigma=sigma, weaken_audio=True)
        eval_train_loader = train_loader
    elif benchmark == "written_spoken_digits_noisy_pairing":
        train_loader, test_loader = digits_get_data_loaders_noisy_pairing(batch_size=batch_size, p=sigma)
        eval_train_loader = train_loader     
    elif benchmark == "nyu_v2_13":
        torch.backends.cudnn.benchmark = True
        train_loader, _, eval_train_loader, test_loader = nyu_v2_get_data_loaders(batch_size=batch_size, num_classes=13, num_workers=2)
    elif benchmark == "nyu_v2_40":
        torch.backends.cudnn.benchmark = True
        train_loader, _, eval_train_loader, test_loader = nyu_v2_get_data_loaders(batch_size=batch_size, num_classes=40, num_workers=2)
    elif benchmark == "mosi":
        train_loader, val_loader, test_loader = mosi_get_data_loaders(batch_size=batch_size)
        eval_train_loader = train_loader
    elif benchmark == "mosei":
        train_loader, val_loader, test_loader = mosei_get_data_loaders(batch_size=batch_size)
        eval_train_loader = train_loader
    else:
        raise ValueError("Invalid benchmark: {}".format(benchmark))
    
    model_name = model
    model, modality = get_model(model, output_dim)
    classifier, criterion = get_classifier_criterion(model_name, model.output_dim, benchmark, eval_train_loader, device)
    model = model.to(device)
    classifier = classifier.to(device)
    if optimizer == "SGD":
        optimizer_type = "SGD"
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "Adam":
        optimizer_type = "Adam"
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")
    # Set up the scheduler if wanted
    if scheduler is not None:
        print("Using scheduler", flush=True) #TODO: remove
        if scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1)
        else:
            raise ValueError("Invalid scheduler")
    # If we want to do supervised training, otherwise continue on to unsupervised training
    if est == "supervised" :
        if hasattr(model, 'give_skips'): # For models that make use of earlier activations 
            model.give_skips = True 
        if benchmark == "nyu_v2_13" or benchmark == "nyu_v2_40":
            model, classifier = supervised_train(model, classifier, criterion, eval_train_loader, device, writer, saver, num_epochs, patience, learning_rate, modality=modality, optimizer_type=optimizer_type, grad_clip=grad_clip, scheduler=scheduler)
            train_loader = eval_train_loader
        elif benchmark == "mosi" or benchmark == "mosei":
            model, classifier = supervised_train(model, classifier, criterion, train_loader, device, writer, saver, num_epochs, patience, learning_rate, modality=modality, optimizer_type=optimizer_type, grad_clip=grad_clip, scheduler=scheduler, valid_loader=val_loader)
            # classifier, criterion = get_classifier_criterion(model_name, model.output_dim, benchmark)
            # classifier = classifier.to(device)
            # model, classifier = eval_train(model, classifier, criterion, optimizer, eval_train_loader, device, writer, saver, eval_lr, eval_num_epochs, eval_patience, modality=modality)
            test(model, classifier, train_loader, device, writer, saver, name='valid', modality=modality)
        elif benchmark == "written_spoken_digits":
            model, classifier = supervised_train(model, classifier, criterion, train_loader, device, writer, saver, num_epochs, patience, learning_rate, modality=modality, optimizer_type=optimizer_type, grad_clip=grad_clip, scheduler=scheduler)
        test(model, classifier, criterion, test_loader, device, writer, saver, name='test', modality=modality)
        test(model, classifier, criterion, train_loader, device, writer, saver, name='train', modality=modality)
        saver.save_collated()
        return model, classifier
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
    model = unsupervised_train(model, optimizer, loss_fun, train_loader, est, temperature, device, writer, saver, num_epochs, patience, modality=modality, grad_clip=grad_clip)
    
    if hasattr(model, 'give_skips'): # For models that make use of earlier activations 
        model.give_skips = True 
    # Now that we have trained the model, we evaluate it by training a linear classifier on top of the frozen representations
    model, classifier = eval_train(model, classifier, criterion, optimizer, eval_train_loader, device, writer, saver, eval_lr, eval_num_epochs, eval_patience, modality=modality, grad_clip=grad_clip)

    # Evaluate the model on the test set
    test(model, classifier, criterion, test_loader, device, writer, saver, 'test', modality=modality)

    # Evaluate the model on the train set
    test(model, classifier, criterion, eval_train_loader, device, writer, saver, 'train', modality=modality)

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
        'learning_rate': 1e-3, 
        'num_epochs': 0.0003,
        'batch_size': 128, #16
        'patience': 10,
        'temperature': 1,
        'output_dim': 1,
        'optimizer': 'SGD',
        'eval_num_epochs': 10,
        'grad_clip': 1.0,
    }
    # Train the model
    model = train(**config)

    # Create save location using slune
    saver = slune.get_csv_saver(config, root_dir='results')
    path = os.path.dirname(saver.getset_current_path())
    # Save the model to the save location using torch.save
    torch.save(model, os.path.join(path, "model.pt"))