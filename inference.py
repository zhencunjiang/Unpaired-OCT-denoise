from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np
import torchvision
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
from math import sqrt




transform = transforms.Compose([
    transforms.ToTensor()
])

class Generator(nn.Module):
    def __init__(self, channels = 1, num_of_layers=10):
        super(Generator, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        self.relu1 = nn.Sigmoid()

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x0):
        out = self.dncnn(x0)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


model = Generator().cuda()
model.load_state_dict(torch.load('/home/ps/zhencunjiang/pycharm_project_773/TCFL-OCT-main/result/saved_models/generator.pth'))
model.eval()


input_folder = '/home/ps/zhencunjiang/pycharm_project_773/OCT_DDPM-main/add_noise/duke17_0.02'
output_folder = '/home/ps/zhencunjiang/pycharm_project_773/OCT_DDPM-main/add_noise_infer/duke17_0.02'


os.makedirs(output_folder, exist_ok=True)
def load_image(filepath):
    image =Image.open(filepath)
    image = np.array(image).astype('float32')/255.0
    return torch.tensor(np.expand_dims(image, axis=0)).unsqueeze(0)

for filename in os.listdir(input_folder):
    if filename.endswith('.tif') or filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        image = Variable(load_image(img_path)).cuda()
        print(image.max())
        with torch.no_grad():
            noise = model(image)
            imgout_test = image - noise
            imgout_test = (imgout_test[0][0]).detach().cpu()
            imgout_test[imgout_test > 1.0] = 1.0
            imgout_test[imgout_test < 0.0] = 0.0

        result_img = imgout_test
        torchvision.utils.save_image(result_img, os.path.join(output_folder, f'result_{filename}'))