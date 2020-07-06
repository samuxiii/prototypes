import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

# torch config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configurations
CHANNELS = 16
WIDTH = HEIGHT = 16 #64
BATCH_LEN = 8
plt.rcParams['toolbar'] = 'None'

###
class Grid:
    def __init__(self):
        super().__init__()
        # Initialization
        self.image = None  # dim: (CHANNELS, WIDTH, HEIGHT)

    def default(self):
        im = Image.open("owl_mini.png") # it should be RGBA
        #im = im.convert('RGBA')
        im = transforms.ToTensor()(im)
        self.load(im)
        return self

    def load(self, img):
        self.image = img[:4,:,:].cpu()
        self.image = torch.clamp(self.image, 0.0, 1.0 )
        return self

    def show(self, title=""):
        img = transforms.ToPILImage()(self.image)
        plt.imshow(img)
        plt.title(title)
        plt.show()

###
# Model is explained here -> https://distill.pub/2020/growing-ca/

class CAModel(nn.Module):
    def __init__(self):
        super().__init__()

        filterY = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).to(device)
        filterX = filterY.t()
        filterId = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]).to(device)

        self.sobelX_kernel = filterX.expand(CHANNELS, 1, 3, 3) / 8.0
        self.sobelY_kernel = filterY.expand(CHANNELS, 1, 3, 3) / 8.0
        self.identity_kernel = filterId.expand(CHANNELS, 1, 3, 3)

        self.fc1 = nn.Conv2d(CHANNELS * 3, 128, 1)
        self.fc2 = nn.Conv2d(128, CHANNELS, 1)
        torch.nn.init.zeros_(self.fc2.weight)  # paper recommendation
        torch.nn.init.zeros_(self.fc2.bias)    # paper recommendation


    def step(self, input, debug=False, showLayers=False):
        # Filters
        sx = F.conv2d(input, self.sobelX_kernel, padding=1, groups=CHANNELS)
        sy = F.conv2d(input, self.sobelY_kernel, padding=1, groups=CHANNELS)
        id = F.conv2d(input, self.identity_kernel, padding=1, groups=CHANNELS)

        # every pixel will have 3*CHANNELS channels now in perception tensor
        perception = torch.cat([id, sx, sy], 1)
        #print(perception.shape)

        x = self.fc1(perception)
        #x = self.norm1(x)
        x = F.relu(x)
        diff = self.fc2(x)  # No relu, we want also negative values, they are the differential image

        # stochastic update for differential image
        stochastic = torch.rand((BATCH_LEN, 1, WIDTH, HEIGHT)) < 0.5
        stochastic = stochastic.type(torch.float).repeat(1,CHANNELS,1,1).to(device)
        #print("stoch:{}".format(stochastic.shape))
        output = input + diff * stochastic  # same tensor will be applied over all the channels

        # alive masking
        alive = F.max_pool2d(output[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1
        alive = alive.type(torch.float).repeat(1,CHANNELS,1,1).to(device)
        #print("alive:{}".format(alive.shape))
        output *= alive

        if showLayers:
            Grid().load(sx[0]).show("Sobel X")
            Grid().load(sy[0]).show("Sobel Y")
            Grid().load(id[0]).show("Identity")
            Grid().load(diff[0]).show("Differential Image")
        if debug or showLayers:
            Grid().load(output[0]).show("Updated Image")

        return output

    def forward(self, input, debug=False, showLayers=False):
        # Chose random steps in between range
        #n_steps = torch.randint(64, 96, (1,))

        # Range of steps to grow up should be within grid dimension
        min_steps = int(WIDTH - WIDTH * 0.2)
        max_steps = int(WIDTH + WIDTH * 0.2)

        n_steps = torch.randint( min_steps, max_steps, (1,))
        output = input.detach().clone().to(device)

        for _ in range(n_steps):
            output = self.step(output, debug, showLayers)
        
        return output

def train(m, origin, target, debug=False):
    target = target.to(device)
    output = origin.to(device)

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(m.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    for epoch in range(10000):
        optimizer.zero_grad()

        output = m.forward(origin)
        loss = loss_f(output[:,:4,...], target[:,:4,...]) # Loss is only calculated with RGBA
        loss.backward()
        torch.nn.utils.clip_grad_norm(m.parameters(), 1) # Prevent gradient to explode (time series problem)

        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            Grid().load(output[0]).show("Epoch {}".format(epoch))
            print("Epoch:{} MSE_Loss:{}".format(epoch, loss))


###
model = CAModel().to(device)
print("model: {} over device={}".format(model, device))
img_target = Grid().default().image
Grid().load(img_target).show("Target")

# Initialize origin image
img_orig = torch.zeros(CHANNELS, WIDTH, HEIGHT)
img_orig[3:, WIDTH//2, HEIGHT//2] = 1.0

# Adding batch dimension
img_orig = img_orig.unsqueeze(0).repeat(BATCH_LEN, 1 , 1, 1)
img_target = img_target.unsqueeze(0).repeat(BATCH_LEN, 1 , 1, 1)

train(model, img_orig, img_target)
