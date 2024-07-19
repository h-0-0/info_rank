import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms

class MNIST_Image_CNN(nn.Module):
    def __init__(self, output_dim=64):
        super(MNIST_Image_CNN, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, self.output_dim) 

    def forward(self, x):
        n_b = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bnorm2(x)
        x = x.view(n_b, -1)
        x = self.fc1(x)
        return x

class MLP(nn.Module):
    """ A multi layer perceptron for representation learning. """
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=64, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Define the layers
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class MNIST_Audio_CNN(nn.Module):
    def __init__(self, output_dim=64):
        super(MNIST_Audio_CNN, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)  
        self.fc1 = nn.Linear(11520, self.output_dim) 

    def forward(self, x):
        n_b = x.shape[0]
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = x.view(n_b, -1) 
        x = self.fc1(x)
        return x

    
class FusionModel(nn.Module):
    def __init__(self, output_dim=64):
        super(FusionModel, self).__init__()

        self.output_dim = output_dim
        
        self.image_encoder = MNIST_Image_CNN(output_dim=output_dim)
        self.audio_encoder = MNIST_Audio_CNN(output_dim=output_dim) 
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, self.output_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 4),
        )

    def forward(self, batch):
        image, audio = batch
        image_repr = self.image_encoder(image)
        audio_repr = self.audio_encoder(audio) 
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def encode_modalities(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        audio_repr = self.audio_encoder(image, audio)
        return image_repr, audio_repr
    
    def fuse(self, image_repr, audio_repr):
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output

    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)
    
class ImageModel(nn.Module):
    def __init__(self, output_dim=64):
        super(ImageModel, self).__init__()

        self.output_dim = output_dim
        
        self.image_encoder = MNIST_Image_CNN(output_dim=output_dim)
        
        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 2),
        )

    def forward(self, batch):
        image_repr = self.image_encoder(batch)
        return image_repr
    
    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)

class AudioModel(nn.Module):
    def __init__(self, output_dim=64):
        super(AudioModel, self).__init__()

        self.output_dim = output_dim
        
        self.audio_encoder = MNIST_Audio_CNN(output_dim=output_dim)
        
        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 2),
        )

    def forward(self, batch):
        audio_repr = self.audio_encoder(batch)
        return audio_repr
    
    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)


class LinearClassifier(nn.Module):
    """ Linear layer we will train as a classifier on top of the representations from the MLP."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define the layer
        self.layer = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        return self.layer(x)
    
if __name__ == '__main__':
    print(MNIST_Image_CNN())
    print(MNIST_Audio_CNN())
    print(FusionModel(64))
    print(LinearClassifier(64, 10))