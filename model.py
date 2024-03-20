import torch.nn as nn
import torch.nn.functional as F
import torch

class MNIST_Image_CNN(nn.Module):
    def __init__(self):
        super(MNIST_Image_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x
    

class MNIST_Audio_CNN(nn.Module):
    def __init__(self):
        super(MNIST_Audio_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(64 * 5 * 4, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x



class FusionModel(nn.Module):
    def __init__(self, output_dim):
        super(FusionModel, self).__init__()
        self.image_encoder = MNIST_Image_CNN()
        self.audio_encoder = MNIST_Audio_CNN()
        
        # Assuming the output dimensions of the image and audio encoders are 50 and 32 respectively
        self.fusion_mlp = nn.Sequential(
            nn.Linear(50 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, image, audio):
        image_repr = self.image_encoder(image)
        audio_repr = self.audio_encoder(audio)
        fused_repr = torch.cat((image_repr, audio_repr), dim=1)
        output = self.fusion_mlp(fused_repr)
        return output

if __name__ == '__main__':
    print(MNIST_Image_CNN())
    print(MNIST_Audio_CNN())
    print(FusionModel(64))