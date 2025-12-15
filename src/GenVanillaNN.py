import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


# 1. Dataset & Transforms
class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(
        self, videoSke, ske_reduced, source_transform=None, target_transform=None
    ):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array(
            [0.5, 0.5, 0.5]
        )
        return denormalized_image


# 2. Advanced Layers (Attention + UpSampling)


class SelfAttention(nn.Module):
    """Self-Attention Layer (SAGAN)"""

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out


class UpBlock(nn.Module):
    """Helper Block: Upsample + Conv (Replaces ConvTranspose2d to reduce noise)"""

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


# New Generator (Smooth U-Net)
class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()

        # Encoder (Downsampling)
        # Input: 3 x 256 x 256
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True)
        )  # -> 128
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> 64
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> 32
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> 16

        # Decoder (Upsampling with UpBlocks)

        # 1. Bottleneck -> 32x32
        self.dec1 = UpBlock(512, 256)

        # 2. Attention
        self.attn = SelfAttention(256)

        # 3. Concatenate with enc3 (256) -> Input 512 -> Output 128
        self.dec2 = UpBlock(512, 128)

        # 4. Concatenate with enc2 (128) -> Input 256 -> Output 64
        self.dec3 = UpBlock(256, 64)

        # 5. Concatenate with enc1 (64) -> Input 128 -> Output 3 (Final Image)
        # Note: The final layer doesn't use UpBlock, standard conv to map to RGB
        self.out_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)  # 64, 128x128
        e2 = self.enc2(e1)  # 128, 64x64
        e3 = self.enc3(e2)  # 256, 32x32
        e4 = self.enc4(e3)  # 512, 16x16

        # Decode
        d1 = self.dec1(e4)  # -> 256, 32x32
        d1 = self.attn(d1)

        # Skip Conn 1
        d1 = torch.cat([d1, e3], dim=1)  # 256+256 = 512

        d2 = self.dec2(d1)  # -> 128, 64x64

        # Skip Conn 2
        d2 = torch.cat([d2, e2], dim=1)  # 128+128 = 256

        d3 = self.dec3(d2)  # -> 64, 128x128

        # Skip Conn 3
        d3 = torch.cat([d3, e1], dim=1)  # 64+64 = 128

        # Final Output
        d_out = self.out_up(d3)  # -> 128, 256x256
        output = self.out_conv(d_out)  # -> 3, 256x256

        return output


class SkeToVectorTransform:
    def __init__(self, ske_reduced=True):
        self.ske_reduced = ske_reduced

    def __call__(self, ske):
        numpy_array = ske.__array__(reduced=self.ske_reduced)
        ske_t = torch.from_numpy(numpy_array).to(torch.float32).flatten()
        ske_t = ske_t.reshape(ske_t.shape[0], 1, 1)
        return ske_t


class GenNNSke26ToImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                self.input_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                1024, 512, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class GenVanillaNN:
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 256
        if optSkeOrImage == 1:
            self.netG = GenNNSke26ToImage()
            src_transform = transforms.Compose([SkeToVectorTransform(ske_reduced=True)])
            self.filename = "data/Dance/DanceGenVanillaFromSke26.pth"
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose(
                [
                    SkeToImageTransform(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.filename = "data/Dance/DanceGenVanillaFromSkeim.pth"

        tgt_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            target_transform=tgt_transform,
            source_transform=src_transform,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=16, shuffle=True
        )
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            try:
                self.netG.load_state_dict(torch.load(self.filename))
            except:
                self.netG = torch.load(self.filename)

    def train(self, n_epochs=20):
        # Keep basic train loop
        pass

    def generate(self, ske):
        self.netG.eval()
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)
        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch)
        res = self.dataset.tensor2image(normalized_output[0])
        return res


if __name__ == "__main__":
    force = False
    optSkeOrImage = (
        1  # use as input a skeleton (1) or an image with a skeleton drawed (2)
    )
    n_epoch = 200
    train = True
    # train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "../data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        # image = image*255
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow("Image", image)
        key = cv2.waitKey(-1)
