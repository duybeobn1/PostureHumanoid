
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # Input : (Batch_size, 3, 64, 64)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.InstanceNorm2d(128, affine=True), #WGAN fits with InstanceNorm rather than BatchNorm
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer : Conv -> 1 chanel (Score)
            # Input : 512 x 4 x 4 -> Output : 1 x 1 x 1
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            # Dont use Sigmoid with WGAN (Critic) , output is Score, not probability [0,1]
        )


    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1) # Output : (Batch_size)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.image_size = 64 
        
        # [Change 1]: Using Generator input is Image (U-Net)
        # instead of GenNNSke26ToImage (vector input)
        self.netG = GenNNSkeImToImage() 
        
        self.netD = Discriminator()

        # These labels are not necessary for WGAN-GP, but it's okay to leave them
        self.real_label = 1.
        self.fake_label = 0.
        
        self.filename = 'data/Dance/DanceGenGAN.pth'

        # Transform cho TARGET (Real image from video) -> [-1, 1]
        tgt_transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.CenterCrop(self.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        
        # [Change 2]: Transform for SOURCE (Skeleton -> Image skeleton) -> [-1, 1]
        src_transform = transforms.Compose([
                            SkeToImageTransform(self.image_size), # Change vector skeleton to image
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

        # [Change 3]: Pass src_transform into Dataset
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, 
                                            target_transform=tgt_transform, 
                                            source_transform=src_transform) # Important!

        # [Change 4]: Add drop_last=True for stability
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, 
                                                      batch_size=32, 
                                                      shuffle=True, 
                                                      drop_last=True)

        # [Change 5]: Update model loading method (Load both G and D)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Loading model from", self.filename)
            checkpoint = torch.load(self.filename)
            # Check if the saved file is a dictionary (new method) or model object (old method)
            if isinstance(checkpoint, dict) and 'netG' in checkpoint:
                self.netG.load_state_dict(checkpoint['netG'])
                self.netD.load_state_dict(checkpoint['netD'])
            else:
                self.netG = checkpoint # Fallback if saved in old way

    def compute_gradient_penalty(self, real_samples, fake_samples, device):
            """Calculates the gradient penalty loss for WGAN GP"""
            # Random weight term for interpolation between real and fake samples
            alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
            
            # Get random interpolation between real and fake samples
            interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
            
            d_interpolates = self.netD(interpolates)
            
            fake = torch.ones(real_samples.size(0), device=device)
            
            # Get gradient w.r.t. interpolates
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty

    def train(self, n_epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GenGAN: Training on {device}")
        
        self.netG.to(device)
        self.netD.to(device)
        
        # Optimizers
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.9))
        
        # Hyperparameters
        lambda_gp = 10      # Gradient penalty weight
        lambda_l1 = 100     # Content loss weight (Important for image keeps the posture correctly)

        print("GenGAN: Starting Training Loop...")
        
        for epoch in range(n_epochs):
            for i, (ske_input, real_img) in enumerate(self.dataloader):
                ske_input = ske_input.to(device)
                real_img = real_img.to(device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizerD.zero_grad()
                
                # Real images
                real_validity = self.netD(real_img)
                
                # Fake images
                fake_img = self.netG(ske_input)
                fake_validity = self.netD(fake_img.detach()) # Detach for not calculate gradient for G this moment
                
                # Gradient Penalty
                gradient_penalty = self.compute_gradient_penalty(real_img, fake_img.detach(), device)
                
                # Loss D (Wasserstein distance)
                # D wants to maximize (Real - Fake) => minimize (Fake - Real)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                
                d_loss.backward()
                optimizerD.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                # Train G each 5 steps of D (standard WGAN-GP), or every step if you want it simpler
                if i % 5 == 0:
                    optimizerG.zero_grad()
                    
                    # Generate again (needs graph gradient)
                    fake_img_g = self.netG(ske_input)
                    fake_validity_g = self.netD(fake_img_g)
                    
                    # Loss G
                    # 1. Adversarial loss: G wants D to think the image is real (maximize output D => minimize -output D)
                    g_adv_loss = -torch.mean(fake_validity_g)
                    # 2. L1 Loss: The generated image must be similar to the real image (pixel-wise)
                    g_l1_loss = F.l1_loss(fake_img_g, real_img)
                    
                    g_loss = g_adv_loss + lambda_l1 * g_l1_loss
                    
                    g_loss.backward()
                    optimizerG.step()
            
            # Print logs
            print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item() if 'g_loss' in locals() else 0:.4f}]")
            
            # Save checkpoint occasionally
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                directory = os.path.dirname(self.filename)
                os.makedirs(directory, exist_ok=True)
                torch.save({
                    'netG': self.netG.state_dict(),
                    'netD': self.netD.state_dict()
                }, self.filename)
                print(f"Saved model to {self.filename}")

    def generate(self, ske):           
        """ generator of image from skeleton """
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.netG.to(device)
        self.netG.eval()

        # Preprocess: Skeleton -> Image -> Tensor 
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0).to(device)  # add batch dimension

        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch)
        
        # to CPU and format for image visualization
        res = self.dataset.tensor2image(normalized_output[0].cpu())
        return res
        # ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        # ske_t = ske_t.to(torch.float32)
        # ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        # normalized_output = self.netG(ske_t)
        # res = self.dataset.tensor2image(normalized_output[0])
        # return res

    

if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "../data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    TRAIN_MODE = True  # True for training, False for loading pre-trained model
    
    if TRAIN_MODE:
        print("Starting training GAN")
        # loadFromFile=False for resetting the model and training from scratch
        gen = GenGAN(targetVideoSke, loadFromFile=True)

        # Train for 200 epochs to achieve decent results
        # Train for 500-1000 epochs to achieve better results
        gen.train(n_epochs=500) 
    else:
        print("Loading pre-trained model...")
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    # Quick test after training
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('GenGAN Test', image)
        key = cv2.waitKey(20) # Wait for 20ms for each frame
        if key == 27: break # Press ESC to exit

