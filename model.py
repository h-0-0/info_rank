import torch.nn as nn
import torch.nn.functional as F
import torch

class MNIST_Image_CNN(nn.Module):
    def __init__(self):
        super(MNIST_Image_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bnorm1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.bnorm2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 64) # TODO change to 128?

        self.W = nn.Parameter(torch.randn(64, 64))

    def forward(self, image, audio):
        x, _ = image, audio
        n_b = x.shape[0]
        x = self.bnorm1(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = self.bnorm2(self.conv2_drop(F.max_pool2d(F.relu(self.conv2(x)),2)))
        x = x.view(n_b, -1)
        x = self.fc1(x)
        return x
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
# TODO: Prev 0.8676

class MNIST_Image_CNN2(nn.Module):
    def __init__(self):
        super(MNIST_Image_CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(0.5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bnorm3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64, 64)

        self.W = nn.Parameter(torch.randn(64, 64))

    def forward(self, image, audio):
        x, _ = image, audio
        n_b = x.shape[0] # batch size
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bnorm2(x)
        x = self.drop2(x)
        x = F.relu(self.conv3(x))
        x = self.bnorm3(x)
        x = x.view(n_b, -1) # flatten
        x = self.fc1(x)
        return x
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
# TODO: Copy of CNN3 but with one more conv layer, Prev 0.8566,

class MNIST_Image_CNN3(nn.Module):
    def __init__(self):
        super(MNIST_Image_CNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 64) 

        self.W = nn.Parameter(torch.randn(64, 64))

    def forward(self, image, audio):
        x, _ = image, audio
        n_b = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bnorm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bnorm2(x)
        x = x.view(n_b, -1)
        x = self.fc1(x)
        return x
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
# TODO: is similar to CNN but with reordering of actfun and pool and dropout remove,
# TODO: 0.9003 in previous run, now trying conv1 with larger channel size (20 instead of 10)
# TODO: got 0.9021, so now increasing channel size of conv2 to 30 from 20
# TODO: got 0.9073, now trying conv1 with channel size 32 (was 20) and conv2 with 64 (was 30)

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
    def __init__(self):
        super(MNIST_Audio_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,1), stride=1)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,1), stride=2)  
        self.fc1 = nn.Linear(4032, 64) 

        self.W = nn.Parameter(torch.randn(64, 64))

    def forward(self, image, audio):
        _, x = image, audio
        n_b = x.shape[0]
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = x.view(n_b, -1) 
        x = self.fc1(x)
        return x

    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
# TODO: Best is 0.2833, now trying with changed dims
# TODO: try more kernels, flipped dim and bigger
# TODO: try a maxpool layer after conv2 aswell as (3,1)
# TODO: try maxpool
# TODO: try dropout
# TODO: try batchnorm

class MNIST_Audio_CNN2(nn.Module):
    def __init__(self):
        super(MNIST_Audio_CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,3), stride=1)  
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=(1,3), stride=2)  
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(640, 64) 

        self.W = nn.Parameter(torch.randn(64, 64))

    def forward(self, image, audio):
        _, x = image, audio
        n_b = x.shape[0]
        x = F.relu(self.conv1(x)) 
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(n_b, -1) 
        x = self.fc1(x)
        return x

    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
# TODO: Prev: 0.2105, now trying with changed dims
# TODO: Try doubling best
# TODO: try batch norm

class MNIST_Audio_CNN3(nn.Module):
    def __init__(self):
        super(MNIST_Audio_CNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)  
        self.pool2 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(5760, 64) 

        self.W = nn.Parameter(torch.randn(64, 64))

    def forward(self, image, audio):
        _, x = image, audio
        n_b = x.shape[0]
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = x.view(n_b, -1) 
        x = self.fc1(x)
        return x

    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
# TODO: CNN but with conv1 with larger channel size, Prev: 0.2349, trying with changed dim.s
# TODO: try rectangular kernel
# TODO: another conv after pool2, followed by maxpool

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.image_encoder = MNIST_Image_CNN()
        # self.audio_encoder = MNIST_Audio_CNN() TODO : Undo
        self.W = nn.Parameter(torch.randn(64, 64))
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, image, audio):
        image_repr = self.image_encoder(image)
        return image_repr
        # audio_repr = self.audio_encoder(audio) TODO: Undo
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
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