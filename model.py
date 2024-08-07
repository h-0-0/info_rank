import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

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


class ResNet101(nn.Module):
    def __init__(self, output_dim=64):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(pretrained=False)
        
        if output_dim != 2048:
            raise ValueError("Currently output dimension must be 2048 for ResNetSegmentation")
        self.output_dim = output_dim
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[1:-2])
        self.rgb_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fusion_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, self.output_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 4),
        )
        
    def forward(self, x):
        x_rgb, x_depth = x
        x_rgb = self.resnet(self.rgb_conv(x_rgb))
        x_depth = self.resnet(self.depth_conv(x_depth))
        x = torch.cat((x_rgb, x_depth), dim=1)
        x = self.fusion_mlp(x)
        return x

    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)
    
class ResNet50(nn.Module):
    def __init__(self, output_dim=64):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        
        if output_dim != 2048:
            raise ValueError("Currently output dimension must be 2048 for ResNetSegmentation")
        self.output_dim = output_dim
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[1:-1])
        self.rgb_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fusion_mlp = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.output_dim),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 4),
        )
        
    def forward(self, x):
        x_rgb, x_depth = x

        x_rgb = self.rgb_conv(x_rgb)
        x_rgb = self.resnet(x_rgb)

        x_depth = self.depth_conv(x_depth)
        x_depth = self.resnet(x_depth)
        
        x = torch.cat((x_rgb, x_depth), dim=1)
        x = self.fusion_mlp(x)
        return x

    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)


class SegClassifier(nn.Module):
    """ To be used for segmentation tasks with ResNetSegmentation. """
    def __init__(self, num_classes):
        super(SegClassifier, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        
    def forward(self, x):
        # Unflatten (N, 2048) -> (N, 2048, 1, 1)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        print(x.shape)
        return x
  
# Could also do: other ResNets, https://huggingface.co/nvidia/mit-b3, https://pytorch.org/vision/stable/models/vision_transformer.html

if __name__ == '__main__':
    print(MNIST_Image_CNN())
    print(MNIST_Audio_CNN())
    print(ImageModel(64))
    print(AudioModel(64))
    print(FusionModel(64))
    print(LinearClassifier(64, 10))
    print(ResNet101(2048))
    print(SegClassifier(13))
    print(models.resnet50(pretrained=False))