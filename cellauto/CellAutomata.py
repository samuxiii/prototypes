import torch
import torch.nn as nn
import torch.optim as optim
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
        self.image = None  # dim: (CHANNELS, WIDTH, HEIGHT)

    def loadIronman(self):
        im = Image.open("owl.png") # it should be RGBA
        #im = im.convert('RGBA')
        im = transforms.ToTensor()(im)
        # Padding will be used to nitialize other channels to zero
        padding = torch.zeros((CHANNELS - im.shape[0], WIDTH, HEIGHT))
        self.image = torch.cat([im, padding], 0)
        return self

    def load(self, img):
        self.image = img
        return self

    def show(self, title=""):
        # img = self.image.permute(1, 2, 0)
        img = transforms.ToPILImage()(self.image[:4, :, :])
        plt.imshow(img)
        plt.title(title)
        plt.show()

###
# Model is explained here -> https://distill.pub/2020/growing-ca/

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
        torch.nn.init.zeros_(self.fc2.weight)  # paper recommendation

    def forward(self, input, debug=False, showLayers=False):
        # Filters
        sx = F.conv2d(input, self.sobelX_kernel, padding=1, groups=CHANNELS).clamp(0., 1.).detach()
        sy = F.conv2d(input, self.sobelY_kernel, padding=1, groups=CHANNELS).clamp(0., 1.).detach()
        id = F.conv2d(input, self.identity_kernel, padding=1, groups=CHANNELS).clamp(0., 1.).detach()

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

        # stochastic update for differential image
        stochastic = torch.rand((1, 1, WIDTH, HEIGHT)) < 0.5
        stochastic = stochastic.type(torch.float)
        #print(stochastic.shape)
        diff = diff * stochastic  # same tensor will be applied over all the channels

        # alive masking
        alive = F.max_pool2d(input[:, 3, :, :], 3, stride=1, padding=1) > 0.1
        alive = alive.type(torch.float)
        #print(alive.shape)

        updated = ((diff + id) * alive).clamp(0., 1.)

        if showLayers:
            Grid().load((sx)[0]).show("Sobel X")
            Grid().load((sy)[0]).show("Sobel Y")
            Grid().load((id)[0]).show("Identity")
            Grid().load((diff)[0]).show("Differential Image")
        if debug or showLayers:
            Grid().load(updated[0]).show("Updated Image")

        return updated


def train(m, origin, target, debug=False):
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(m.parameters(), lr=0.01)
    target.unsqueeze_(0)

    # Chose random steps in between range
    n_steps = torch.randint(64, 96, (1,))

    for i in range(n_steps):
        optimizer.zero_grad()
        origin = m.forward(origin, debug)
        loss = loss_f(origin, target)
        print("({}/{}) MSE Loss: {}".format(i+1, n_steps, loss))
        loss.backward()
        optimizer.step()

        Grid().load(origin[0]).show()


###
model = CAModel()
print("model: {}".format(model))
img_target = Grid().loadIronman().image

# Initialize origin image
img_orig = torch.zeros_like(img_target)
img_orig[3:, WIDTH//2, HEIGHT//2] = 1.0
#Grid().load(img_orig).show("Origin")

# Adding batch dimension to img_orig
img_orig.unsqueeze_(0)
train(model, img_orig, img_target)