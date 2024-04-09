import torch.nn as nn
import torch.nn.functional as F
import torch

class MNIST_Image_CNN(nn.Module):
    def __init__(self):
        super(MNIST_Image_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 128)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x


class MNIST_Audio_CNN(nn.Module):
    def __init__(self):
        super(MNIST_Audio_CNN, self).__init__()
        
        # Input shape: (batch_size, 39, 13)
        self.conv1 = nn.Conv1d(in_channels=13, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512, 128)  # Adjust input size based on output shape after convolutions
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        
        return x

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.image_encoder = MNIST_Image_CNN()
        self.audio_encoder = MNIST_Audio_CNN()
        self.W = nn.Parameter(torch.randn(64, 64))
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, image, audio):
        image_repr = self.image_encoder(image)
        audio_repr = self.audio_encoder(audio)
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t()))  #/ temperature
    # TODO: Temperature applied correctly? / Should maybe get rid of it?

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