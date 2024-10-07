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
        self.num_classes = output_dim
        
        # Define the layer
        self.layer = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        return self.layer(x)


class ResNet101(nn.Module):
    def __init__(self, output_dim=64):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(weights=None)
        self.encode_batch = True # When True loss function can be applied to encoded modalities
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

        x_rgb = self.rgb_conv(x_rgb)
        x_rgb = self.resnet(x_rgb)

        # x_rgb = torch.utils.checkpoint.checkpoint(self.rgb_conv, x_rgb, use_reentrant=False)
        # x_rgb = torch.utils.checkpoint.checkpoint(self.resnet, x_rgb, use_reentrant=False)

        x_depth = self.depth_conv(x_depth)
        x_depth = self.resnet(x_depth)

        # x_depth = torch.utils.checkpoint.checkpoint(self.depth_conv, x_depth, use_reentrant=False)
        # x_depth = torch.utils.checkpoint.checkpoint(self.resnet, x_depth, use_reentrant=False)

        x = torch.cat((x_rgb, x_depth), dim=1)
        x = self.fusion_mlp(x)
        return x
    
    def encode_modalities(self, x):
        x_rgb, x_depth = x
  
        x_rgb = self.rgb_conv(x_rgb)
        x_rgb = self.resnet(x_rgb)

        x_depth = self.depth_conv(x_depth)
        x_depth = self.resnet(x_depth)

        return [x_rgb, x_depth]
    
    def fuse(self, x):
        x_rgb, x_depth = x 
        x = torch.cat((x_rgb, x_depth), dim=1)
        x = self.fusion_mlp(x)
        return x

    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)
    
class ResNet50(nn.Module):
    def __init__(self, output_dim=64):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.encode_batch = True # When True loss function can be applied to encoded modalities
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

        # x_rgb = torch.utils.checkpoint.checkpoint(self.rgb_conv, x_rgb, use_reentrant=False)
        # x_rgb = torch.utils.checkpoint.checkpoint(self.resnet, x_rgb, use_reentrant=False)

        x_depth = self.depth_conv(x_depth)
        x_depth = self.resnet(x_depth)

        # x_depth = torch.utils.checkpoint.checkpoint(self.depth_conv, x_depth, use_reentrant=False)
        # x_depth = torch.utils.checkpoint.checkpoint(self.resnet, x_depth, use_reentrant=False)
        
        x = torch.cat((x_rgb, x_depth), dim=1)
        x = self.fusion_mlp(x)
        return x
    
    def encode_modalities(self, x):
        x_rgb, x_depth = x
  
        x_rgb = self.rgb_conv(x_rgb)
        x_rgb = self.resnet(x_rgb)

        x_depth = self.depth_conv(x_depth)
        x_depth = self.resnet(x_depth)

        return [x_rgb, x_depth]
    
    def fuse(self, x):
        x_rgb, x_depth = x 
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
        self.num_classes = num_classes
        self.decoder = nn.Sequential(
            # First transposed convolution layer
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=2, padding=0),  # Output: (N, 1024, 8, 8)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=0),  # Output: (N, 1024, 8, 8)
            nn.ReLU(inplace=True),
            # # Second transposed convolution layer
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3,5), stride=2, padding=0),  # Output: (N, 512, 22, 22)
            nn.ReLU(inplace=True),

            # # Third transposed convolution layer
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,5), stride=2, padding=0),  # Output: (N, 256, 50, 50)
            nn.ReLU(inplace=True),

            # # Fourth transposed convolution layer
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3,5), stride=2, padding=0),  # Output: (N, 128, 106, 106)
            nn.ReLU(inplace=True),

            # # Fifth transposed convolution layer
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,5), stride=2, padding=0),   # Output: (N, 64, 218, 218)
            nn.ReLU(inplace=True),

            # Sixth transposed convolution layer
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3,5), stride=2, padding=0),   # Output: (N, 32, 64, 64)
            nn.ReLU(inplace=True),

            # Seventh transposed convolution layer
            nn.ConvTranspose2d(in_channels=16, out_channels=num_classes, kernel_size=2, stride=1, padding=0),   # Output: (N, 32, 64, 64)
            nn.ReLU(inplace=True),
        )        
        
    def forward(self, x):
        # Unflatten (N, 2048) -> (N, 2048, 1, 1)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)
        x = x[:, :, :240, :320]
        return x
  
# TODO: Could also do: other ResNets, https://huggingface.co/nvidia/mit-b3, https://pytorch.org/vision/stable/models/vision_transformer.html

class MosiFusion(nn.Module):
    """Fusion model for MOSI dataset.
    Consists of three GRU encoders for vision, audio and text modalities.
    The output of each encoder is concatenated and passed through an MLP.
    """
    def __init__(self, output_dim=64):
        super(MosiFusion, self).__init__()

        self.output_dim = output_dim

        # self.image_encoder = nn.LSTM(input_size=35, hidden_size=256, num_layers=2, batch_first=True)
        # self.audio_encoder = nn.LSTM(input_size=74, hidden_size=256, num_layers=2, batch_first=True)
        # self.text_encoder = nn.LSTM(input_size=300, hidden_size=256, num_layers=2, batch_first=True)

        hidden_size = 512

        self.image_encoder = nn.GRU(input_size=35, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.image_norm = nn.LayerNorm(hidden_size*2)
        self.image_proj = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size*50*2, hidden_size)
        )

        self.audio_encoder = nn.GRU(input_size=74, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.audio_norm = nn.LayerNorm(hidden_size*2)
        self.audio_proj = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size*50*2, hidden_size)
        )

        self.text_encoder = nn.GRU(input_size=300, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.text_norm = nn.LayerNorm(hidden_size*2)
        self.text_proj = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size*50*2, hidden_size)
        )

        # self.fusion_mlp = nn.Sequential(
        #     nn.BatchNorm1d(hidden_size*3),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size*3, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size, self.output_dim)
        # )

        self.fusion_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_dim),
        )

        # self.fusion_mlp = nn.Sequential(
        #     nn.BatchNorm1d(hidden_size*3),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size*3, hidden_size*2),
        #     nn.BatchNorm1d(hidden_size*2),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size, self.output_dim)
        # )

        # self.fusion_mlp = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(hidden_size*3, hidden_size*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, self.output_dim),
        # )

        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 8),
        )

    def forward(self, batch):
        image, audio, text = batch

        image_repr, _ = self.image_encoder(image)
        image_repr = self.image_norm(image_repr)
        image_repr = self.image_proj(torch.flatten(image_repr, start_dim=1))

        audio_repr, _ = self.audio_encoder(audio)
        audio_repr = self.audio_norm(audio_repr)
        audio_repr = self.audio_proj(torch.flatten(audio_repr, start_dim=1))

        text_repr, _ = self.text_encoder(text)
        text_repr = self.text_norm(text_repr)
        text_repr = self.text_proj(torch.flatten(text_repr, start_dim=1))

        fused_repr = torch.cat((image_repr, audio_repr, text_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)
        
class MoseiFusion(nn.Module):
    """Fusion model for MOSEI dataset.
    Consists of three GRU encoders for vision, audio and text modalities.
    The output of each encoder is concatenated and passed through an MLP.
    """
    def __init__(self, output_dim=64):
        super(MoseiFusion, self).__init__()

        self.output_dim = output_dim

        # self.image_encoder = nn.LSTM(input_size=713, hidden_size=256, num_layers=2, batch_first=True)
        # self.audio_encoder = nn.LSTM(input_size=74, hidden_size=256, num_layers=2, batch_first=True)
        # self.text_encoder = nn.LSTM(input_size=300, hidden_size=256, num_layers=2, batch_first=True)

        hidden_size = 256

        self.image_encoder = nn.GRU(input_size=713, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.image_norm = nn.LayerNorm(hidden_size*2)
        self.image_proj = nn.Sequential(
            nn.Linear(hidden_size*50*2, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

        self.audio_encoder = nn.GRU(input_size=74, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.audio_norm = nn.LayerNorm(hidden_size*2)
        self.audio_proj = nn.Sequential(
            nn.Linear(hidden_size*50*2, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

        self.text_encoder = nn.GRU(input_size=300, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.text_norm = nn.LayerNorm(hidden_size*2)
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size*50*2, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )

        self.fusion_mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_size*3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size*3, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, self.output_dim),
            nn.BatchNorm1d(self.output_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.output_dim*2, 8),
        )

    def forward(self, batch):
        image, audio, text = batch
        # Swap axes 1 and 2
        # image = image.permute(0, 2, 1) #TODO Remove
        # image_repr = self.image_downsample(image)#TODO Remove
        # image_repr = image_repr.permute(0, 2, 1) #TODO Remove
        image_repr, _ = self.image_encoder(image)
        image_repr = self.image_norm(image_repr)
        image_repr = self.image_proj(torch.flatten(image_repr, start_dim=1))
    
        audio_repr, _ = self.audio_encoder(audio)
        audio_repr = self.audio_norm(audio_repr)
        audio_repr = self.audio_proj(torch.flatten(audio_repr, start_dim=1))

        text_repr, _ = self.text_encoder(text)
        text_repr = self.text_norm(text_repr)
        text_repr = self.text_proj(torch.flatten(text_repr, start_dim=1))

        fused_repr = torch.cat((image_repr, audio_repr, text_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output
    
    def score(self, u, v):
        concat = torch.cat((u, v), dim=1)
        return self.critic(concat)

# https://github.com/declare-lab/MISA/blob/master/src/models.py#L47
# Could try LSTM
# use hidden size equal to input dimension size and output dimension equal to the number of classes
# use two sets of LSTM/GRU's 
# Could add dropout to GRU
# Could add batch normalization or layer normalization
# Adam may increase stability
# Could add L2 regularization
# Before MLP projection could add avg pooling
# Change activation functions from Relu to Tanh

class Regression(nn.Module):
    """ TODO """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.num_classes = 1
        
        # Define MLP
        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.input_dim//4),
            nn.ReLU(),
            nn.Linear(self.input_dim//4, self.output_dim)
        )
        # self.layer = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    print(MNIST_Image_CNN())
    print(MNIST_Audio_CNN())
    print(ImageModel(64))
    print(AudioModel(64))
    print(FusionModel(64))
    print(LinearClassifier(64, 10))
    print(ResNet101(2048))
    print(SegClassifier(13))
    print(ResNet50(2048))