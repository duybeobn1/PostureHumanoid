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

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output


# Self attention custom layer
class SelfAttention(nn.Module):
    """ Self-Attention Layer (SAGAN) """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        # Query, Key, Value
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Gamma: parameters learnt for balance attention and original feature map
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSke26ToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton_dim26)->Image
    """
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            # Entrée: (Batch, 26, 1, 1) -> Sortie: (Batch, 1024, 4, 4)
            nn.ConvTranspose2d(self.input_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            # (Batch, 1024, 4, 4) -> Sortie: (Batch, 512, 8, 8)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # (Batch, 512, 8, 8) -> Sortie: (Batch, 256, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # (Batch, 256, 16, 16) -> Sortie: (Batch, 128, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # (Batch, 128, 32, 32) -> Sortie: (Batch, 3, 64, 64)
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Tanh pour que la sortie soit dans la plage [-1, 1] comme la cible (tgt_transform)
        )
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        return img



class GenNNSkeImToImage(nn.Module):
    """ Generator: Skeleton Image -> Real Image (U-Net like architecture) """
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        
        # Decoder (Upsampling)
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        
        # Attention layer
        self.attn = SelfAttention(256)
        
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.out = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, x):
        # Encoding
        e1 = self.enc1(x) # 32x32
        e2 = self.enc2(e1) # 16x16
        e3 = self.enc3(e2) # 8x8
        e4 = self.enc4(e3) # 4x4
        
        
        # Decoding
        d1 = self.dec1(e4) # 8x8
        # Attention
        d1 = self.attn(d1)

        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.dec2(d1) # 128 channels, 16x16    
        d2 = torch.cat([d2, e2], dim=1)

        # Dec 3: Concat d2 (128ch, 16x16) and e2 (128ch, 16x16) -> 256 total channels
        d3 = self.dec3(d2) # 64 channels, 32x32  
        d3 = torch.cat([d3, e1], dim=1)

        # Output: Concat d3 (64ch, 32x32) and e1 (64ch, 32x32) -> 128 total channels
        output = self.out(d3) # 64x64
        return output


class SkeToVectorTransform:
    """ Convertit l'objet Skeleton en un tenseur PyTorch (D, 1, 1) pour le réseau. """
    def __init__(self, ske_reduced=True):
        self.ske_reduced = ske_reduced

    def __call__(self, ske):
        # ske.__array__(reduced=True) retourne le tableau NumPy (13, 2)
        numpy_array = ske.__array__(reduced=self.ske_reduced)
        
        ske_t = torch.from_numpy(numpy_array).to(torch.float32).flatten()
        
        ske_t = ske_t.reshape(ske_t.shape[0], 1, 1)
        
        return ske_t



class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 256
        if optSkeOrImage==1:        # skeleton_dim26 to image
            self.netG = GenNNSke26ToImage()
            src_transform = transforms.Compose([ SkeToVectorTransform(ske_reduced=True)])
            self.filename = 'data/Dance/DanceGenVanillaFromSke26.pth'
        else:                       # skeleton_image to image
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'



        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            # ouput image (target) are in the range [-1,1] after normalization
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):
        criterion = nn.MSELoss() 
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        print("GenVanillaNN: Starting Training Loop...")
        directory = os.path.dirname(self.filename)
        os.makedirs(directory, exist_ok=True)
        print(f"Target save directory checked/created: {directory}")

        self.netG.train()

        for epoch in range(n_epochs):
            for i, (source_data, target_image) in enumerate(self.dataloader):
                optimizer.zero_grad()

                generated_image = self.netG(source_data)
                
                loss = criterion(generated_image, target_image)
                loss.backward()
                optimizer.step()

                # Affichage des métriques d'entraînement
                if i % 100 == 0:
                    print(f"[{epoch+1}/{n_epochs}][{i}/{len(self.dataloader)}] Loss: {loss.item():.4f}")

        
        torch.save(self.netG.state_dict(), self.filename)
        print(f"Model saved state_dict to {self.filename}")

        print("GenVanillaNN: Training Finished.")


    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
    
        ske_t = self.dataset.preprocessSkeleton(ske)
        # Prepa du tenseur pour le réseau (ajout de la dimension du batch)
        ske_t_batch = ske_t.unsqueeze(0)   # make a batch
        
        with torch.no_grad(): 
            normalized_output = self.netG(ske_t_batch)
        
        res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res




if __name__ == '__main__':
    force = False
    optSkeOrImage = 2           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 200  
    train = True 
    #train = True

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
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
