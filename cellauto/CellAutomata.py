import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Configurations
CHANNELS = 16
WIDTH = HEIGHT = 64
plt.rcParams['toolbar'] = 'None'

###
class Grid:
    def __init__(self):
        super().__init__()
        # Initialization
        self.image = torch.zeros((CHANNELS, WIDTH, HEIGHT))

    def loadIronman(self):
        im = Image.open("ironman.jpg")
        self.image = transforms.ToTensor()(im)
        return self

    def load(self, img):
        self.image = img
        return self

    def show(self, title=""):
        # img = self.image.permute(1, 2, 0)
        img = transforms.ToPILImage()(self.image[:3, :, :])
        plt.imshow(img)
        plt.title(title)
        plt.show()


# Grid().loadIronman().show()

###
# Model obtrained from -> https://distill.pub/2020/growing-ca/

class CAModel(nn.Module):
    def __init__(self):
        super().__init__()

        filterY = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        filterX = filterY.t()
        filterId = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])

        self.sobelX_kernel = filterX.expand(CHANNELS, 1, 3, 3)
        self.sobelY_kernel = filterY.expand(CHANNELS, 1, 3, 3)
        self.identity_kernel = filterId.expand(CHANNELS, 1, 3, 3)

        self.fc1 = nn.Linear(CHANNELS * 3, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, inputs, showLayers=False):
        # inputs = torch.tensor(minibatch, CHANNELS, WIDTH, HEIGHT)
        sx = F.conv2d(inputs, self.sobelX_kernel, padding=1, groups=CHANNELS).clamp(0., 1.)
        sy = F.conv2d(inputs, self.sobelY_kernel, padding=1, groups=CHANNELS).clamp(0., 1.)
        id = F.conv2d(inputs, self.identity_kernel, padding=1, groups=CHANNELS).clamp(0., 1.)

        # every pixel will have 48 channels now in perception tensor
        # shape: [batch, 48, 64, 64]
        perception = torch.cat([sx, sx, id], 1)
        #print(perception.shape)
        #print(perception[0,:,32,32])

        # Now we need to pass every pixel through a dense layer
        # so reshaping is needed: [1, 48, 64, 64] -> [batch_pixels, 48]
        perception = perception.permute(0, 2, 3, 1).squeeze(0).view(WIDTH * HEIGHT, CHANNELS * 3)
        #print(perception.shape)
        #print(perception[32 + 32*HEIGHT])

        x = self.fc1(perception)
        x = F.relu(x)
        x = self.fc2(x)  # No relu, we want also negative values, they are the differential image

        # Reshaping again to get the differential image
        # so [batch_pixels, 16] -> [1, 16, 64, 64]
        diff = x.view(1, WIDTH, HEIGHT, CHANNELS).permute(0, 3, 1, 2)
        #print(diff.shape)

        if showLayers:
            Grid().load((sx)[0]).show("Sobel X")
            Grid().load((sy)[0]).show("Sobel Y")
            Grid().load((id)[0]).show("Identity")
            Grid().load((diff)[0]).show("Differential Image")

        return diff

###
m = CAModel()
img = Grid().loadIronman().image
padding = torch.zeros((CHANNELS - 3, WIDTH, HEIGHT))
img = torch.cat([img, padding], 0).unsqueeze(0)  # Initialize other channels to zero
x = m.forward(img, True)

#for _ in range(5):
#    x = m.forward(x, True)
