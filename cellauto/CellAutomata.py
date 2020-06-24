import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Configurations
CHANNELS = 16
WIDTH = HEIGHT = 64

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
        #img = self.image.permute(1, 2, 0)
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

        filterY = torch.tensor([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]])
        filterX = filterY.t()
        filterId = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])

        self.sobelX_kernel = filterX.expand(CHANNELS, 1, 3, 3)
        self.sobelY_kernel = filterY.expand(CHANNELS, 1, 3, 3)
        self.identity_kernel = filterId.expand(CHANNELS, 1, 3, 3)

        self.fc1 = nn.Linear(CHANNELS * 3, 128)
        self.fc2 = nn.Linear(128, 16)
  
    def forward(self, inputs):
        # inputs = torch.tensor(minibatch, CHANNELS, WIDTH, HEIGHT)
        sx = F.conv2d(inputs, self.sobelX_kernel, padding=1, groups=CHANNELS).clamp(0., 1.)
        sy = F.conv2d(inputs, self.sobelY_kernel, padding=1, groups=CHANNELS).clamp(0., 1.)
        id = F.conv2d(inputs, self.identity_kernel, padding=1, groups=CHANNELS).clamp(0., 1.)

        Grid().load((sx)[0]).show("Sobel X")
        Grid().load((sy)[0]).show("Sobel Y")
        Grid().load((id)[0]).show("Identity")


m = CAModel()
img = Grid().loadIronman().image
padding = torch.zeros((CHANNELS - 3, WIDTH, HEIGHT))
img = torch.cat([img, padding], 0).unsqueeze(0) # Initialize other channels to zero
m.forward(img)