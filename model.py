import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms

class MNIST_Image_CNN(nn.Module):
    def __init__(self):
        super(MNIST_Image_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 128) 

        self.output_dim = 128

        self.W = nn.Parameter(torch.randn(128, 128))

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)  
        self.fc1 = nn.Linear(11520, 128) 

        self.output_dim = 128

        self.W = nn.Parameter(torch.randn(128, 128))

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

    
class FusionModel(nn.Module):
    def __init__(self, single_modality=None):
        super(FusionModel, self).__init__()

        self.output_dim = 128
        
        self.image_encoder = MNIST_Image_CNN()
        self.audio_encoder = MNIST_Audio_CNN() 
        if single_modality == 'image':
            self.forward = self.forward_image_only
        elif single_modality == 'audio':
            self.forward = self.forward_audio_only
        else: 
            self.forward = self.forward_both

        self.W = nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

    def forward_image_only(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        audio_repr = torch.zeros(audio.shape[0], self.output_dim, device=audio.get_device())
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def forward_audio_only(self, image, audio):
        image_repr = torch.zeros(image.shape[0], self.output_dim, device=image.get_device())
        audio_repr = self.audio_encoder(image, audio) 
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output

    def forward_both(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        audio_repr = self.audio_encoder(image, audio) 
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
    # TODO: Temperature applied correctly? / Should maybe get rid of it?

class FusionModelStrict(nn.Module):
    def __init__(self, single_modality=None):
        super(FusionModelStrict, self).__init__()

        self.output_dim = 128
        
        self.image_encoder = MNIST_Image_CNN()
        self.audio_encoder = MNIST_Audio_CNN() 
        if single_modality == 'image':
            self.forward = self.forward_image_only
            input_dim = 128
        elif single_modality == 'audio':
            self.forward = self.forward_audio_only
            input_dim = 128
        else: 
            self.forward = self.forward_both
            input_dim = 256

        self.W = nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

    def forward_image_only(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        output = self.fusion_mlp(image_repr)
        return output
    
    def forward_audio_only(self, image, audio):
        audio_repr = self.audio_encoder(image, audio) 
        output = self.fusion_mlp(audio_repr)
        return output

    def forward_both(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        audio_repr = self.audio_encoder(image, audio) 
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def score(self, u, v, temperature):
        return torch.matmul(u, torch.matmul(self.W, v.t())) / temperature
    # TODO: Temperature applied correctly? / Should maybe get rid of it?

class FusionModelStrictShallow(nn.Module):
    def __init__(self, single_modality=None):
        super(FusionModelStrictShallow, self).__init__()

        self.output_dim = 128
        
        self.image_encoder = MNIST_Image_CNN()
        self.audio_encoder = MNIST_Audio_CNN() 
        if single_modality == 'image':
            self.forward = self.forward_image_only
            input_dim = 128
        elif single_modality == 'audio':
            self.forward = self.forward_audio_only
            input_dim = 128
        else: 
            self.forward = self.forward_both
            input_dim = 256

        self.W = nn.Parameter(torch.randn(self.output_dim, self.output_dim))
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, self.output_dim),
        )

    def forward_image_only(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        output = self.fusion_mlp(image_repr)
        return output
    
    def forward_audio_only(self, image, audio):
        audio_repr = self.audio_encoder(image, audio) 
        output = self.fusion_mlp(audio_repr)
        return output

    def forward_both(self, image, audio):
        image_repr = self.image_encoder(image, audio)
        audio_repr = self.audio_encoder(image, audio) 
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