from pathlib import Path
from os import makedirs
import torch
import tqdm

def make_position_vectors(width, height):
    xs = torch.linspace(0, 1, steps=width)
    ys = torch.linspace(0, 1, steps=height)
    # x, y = torch.meshgrid(xs, ys, indexing='xy')
    x, y = torch.meshgrid(xs, ys)
    return torch.stack([x, y])

import math
from torch import nn
from torch.nn import functional as F

class MultiresolutionHashEncoder2d(nn.Module):
    def __init__(self, l=16, t=2**14, f=2, n_min=16, n_max=512, interpolation='bilinear'):
        super().__init__()
        self.l = l
        self.t = t
        self.f = f
        self.interpolation = interpolation

        b = math.exp((math.log(n_max) - math.log(n_min)) / (l - 1))
        self.ns = [int(n_min * (b ** i)) for i in range(l)]

        # Prime Numbers from https://github.com/NVlabs/tiny-cuda-nn/blob/ee585fa47e99de4c26f6ae88be7bcb82b9295310/include/tiny-cuda-nn/encodings/grid.h
        self.register_buffer('primes', torch.tensor([1, 2654435761]))
        
        self.hash_table = nn.Parameter(
            torch.rand([l, t, f], requires_grad=True) * 2e-4 - 1e-4)

    @property
    def encoded_vector_size(self):
        return self.l * self.f
        
    def forward(self, x):
        b, c, h, w = x.size()

        def make_grid(x, n):
            g = F.max_pool2d(x * n, (h // n, w // n), stride=1).to(dtype=torch.long)
            g = g * self.primes.view([2, 1, 1])
            g = (g[:,0] ^ g[:,1]) % self.t
            return g

        grids = [make_grid(x, n) for n in self.ns]
        features = [self.hash_table[i, g].permute(0, 3, 1, 2)
                    for i, g in enumerate(grids)]
        feature_map = torch.hstack([
            F.interpolate(f, (h, w), mode=self.interpolation)
            for f in features
        ]) 

        return feature_map
    
class Model(nn.Module):
    def __init__(self, encoder, num_planes=64, num_layers=2):
        super().__init__()
        self.enc = encoder
        
        # 1x1 convolution is equivalent to MLP for a point in the 2D-coordinates
        layers = [nn.Conv2d(encoder.encoded_vector_size, num_planes, 1)]
        for _ in range(num_layers - 2):
            layers += [nn.ReLU(),
                       nn.Conv2d(num_planes, num_planes, 1)]
        layers += [nn.ReLU(),
                   nn.Conv2d(num_planes, 3, 1),
                   nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.enc(x)
        out = self.mlp(feature)
        return out


def save_image(x, name):
    p = Path(name).parents[0]

    if not p.is_dir():
        makedirs(p)
    
    img = (x * 255).to(torch.uint8)
    write_png(img, name + '.png')



def train(name, img, model, mask=None, steps=300, output_visualize=True):
    _, w, h = img.size()
    x = make_position_vectors(w, h)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam([
        { 'params': model.enc.parameters() },
        { 'params': model.mlp.parameters(), 'weight_decay': 1e-6 }
    ], lr=0.01, betas=(0.9, 0.99), eps=1e-15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    x, y = x.to(device), img.to(device)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    
    if mask is not None:
        mask = mask.to(device)
    
    for i in tqdm.tqdm(range(steps)):
        pred = model(x)

        if mask is None:
            loss = loss_fn(pred, y)
        else:
            loss = loss_fn(pred * mask, y * mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if output_visualize:
            save_image(pred[0].cpu(), f"outputs/{name}/{i:010d}")
    print(f"step: {i}, loss: {loss.item()}")

from torchvision.io import read_image, ImageReadMode, write_png

# img = read_image('data/image/tokyo.jpg', ImageReadMode.RGB)
img = read_image('data/image/tokyo.bin', ImageReadMode.RGB)

img = img / 255

model = Model(encoder=MultiresolutionHashEncoder2d())
train('tokyo_bin', img, model)
