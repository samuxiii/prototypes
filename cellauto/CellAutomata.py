import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib notebook

from torchvision import transforms
from PIL import Image

# torch config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations
CHANNELS = 16
WIDTH = HEIGHT = 64
plt.rcParams['toolbar'] = 'None'


# plt.ion()

###
class Grid:
    def __init__(self):
        super().__init__()
        # Initialization
        self.image = None  # dim: (CHANNELS, WIDTH, HEIGHT)

    def loadIronman(self):
        im = Image.open("owl.png")  # it should be RGBA
        # im = im.convert('RGBA')
        im = transforms.ToTensor()(im)
        # Padding will be used to initialize other channels to zero
        # padding = torch.zeros((CHANNELS - im.shape[0], WIDTH, HEIGHT))
        # self.load(torch.cat([im, padding], 0))
        self.load(im)
        return self

    def load(self, img):
        self.image = img.cpu()
        return self

    def show(self, title=""):
        # img = self.image.permute(1, 2, 0)
        img = transforms.ToPILImage()(self.image[:4, :, :])
        plt.imshow(img)
        plt.title(title)
        plt.show()
        # plt.draw()


###
# Model is explained here -> https://distill.pub/2020/growing-ca/

class CAModel(nn.Module):
    def __init__(self):
        super().__init__()

        filterY = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).to(device)
        filterX = filterY.t()
        filterId = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]).to(device)

        self.sobelX_kernel = filterX.expand(CHANNELS, 1, 3, 3)
        self.sobelY_kernel = filterY.expand(CHANNELS, 1, 3, 3)
        self.identity_kernel = filterId.expand(CHANNELS, 1, 3, 3)

        self.fc1 = nn.Conv2d(48, 128, 1)
        self.fc2 = nn.Conv2d(128, 16, 1)
        torch.nn.init.zeros_(self.fc2.weight)  # paper recommendation
        torch.nn.init.zeros_(self.fc2.bias)  # paper recommendation

    def step(self, input, debug=False, showLayers=False):
        BATCH_LEN = input.shape[0]
        # Filters
        sx = F.conv2d(input, self.sobelX_kernel, padding=1, groups=CHANNELS).detach()
        sy = F.conv2d(input, self.sobelY_kernel, padding=1, groups=CHANNELS).detach()
        id = F.conv2d(input, self.identity_kernel, padding=1, groups=CHANNELS).detach()

        # every pixel will have 48 channels now in perception tensor
        # shape: [batch, 48, 64, 64]
        perception = torch.cat([id, sx, sy], 1)
        # print(perception.shape)

        x = self.fc1(perception)
        # x = self.norm1(x)
        x = F.relu(x)
        diff = self.fc2(x)  # No relu, we want also negative values, they are the differential image

        # stochastic update for differential image
        stochastic = torch.rand((BATCH_LEN, 1, WIDTH, HEIGHT)) < 0.5
        stochastic = stochastic.type(torch.float).repeat(1, 16, 1, 1).to(device)
        # print("stoch:{}".format(stochastic.shape))
        output = input + diff * stochastic  # same tensor will be applied over all the channels

        # alive masking
        alive = F.max_pool2d(output[:, 3, :, :], 3, stride=1, padding=1) > 0.1
        alive = alive.type(torch.float).view(BATCH_LEN, 1, 64, 64).repeat(1, 16, 1, 1)
        # print("alive:{}".format(alive.shape))
        output *= alive

        if showLayers:
            Grid().load((sx)[0]).show("Sobel X")
            Grid().load((sy)[0]).show("Sobel Y")
            Grid().load((id)[0]).show("Identity")
            Grid().load((diff)[0]).show("Differential Image")
        if debug or showLayers:
            Grid().load(output[0]).show("Updated Image")

        return torch.clamp(output, 0.0, 1.0)

    def forward(self, input, debug=False, showLayers=False):
        # Chose random steps in between range
        n_steps = torch.randint(64, 96, (1,))
        output = input.to(device)

        for _ in range(n_steps):
            output = self.step(output)

        return output


def train(m, origin, target, debug=False):
    target = target.to(device)

    for epoch in range(1000):
        loss_f = nn.MSELoss()
        optimizer = optim.Adam(m.parameters(), lr=0.0001)
        optimizer.zero_grad()

        output = m.forward(origin)

        loss = loss_f(output[:, :4, ...], target[:, :4, ...])  # Loss is only calculated with RGBA
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            Grid().load(output[0]).show("Epoch {}".format(epoch))
            print("Epoch:{} MSE_Loss:{}".format(epoch, loss))


###
model = CAModel().to(device)
print("model: {} over device={}".format(model, device))
img_target = Grid().loadIronman().image

# Initialize origin image
img_orig = torch.zeros(CHANNELS, WIDTH, HEIGHT)
img_orig[3:, WIDTH // 2, HEIGHT // 2] = 1.0
# Grid().load(img_orig).show("Origin")

# Adding batch dimension
img_orig.unsqueeze_(0)
img_orig = img_orig.repeat(1, 1, 1, 1)
img_target.unsqueeze_(0)
img_target = img_target.repeat(1, 1, 1, 1)

train(model, img_orig, img_target)