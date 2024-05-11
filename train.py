import argparse
import torch
import model
import cv2 as cv
import HeadPoseEstimation
import vgg_face
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from EmoDataset import EMODataset
import torch.nn.functional as F
import decord
from omegaconf import OmegaConf
from torchvision import models
from model import GazeLoss,Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


'''
Perceptual Loss:

The PerceptualLoss class combines losses from VGG19, VGG Face, and a specialized gaze loss.
It computes the perceptual losses by passing the output and target frames through the respective models and calculating the MSE loss between the features.
The total perceptual loss is a weighted sum of the individual losses.


Adversarial Loss:

The adversarial_loss function computes the adversarial loss for the generator.
It passes the generated output frame through the discriminator and calculates the MSE loss between the predicted values and a tensor of ones (indicating real samples).


Cycle Consistency Loss:

The cycle_consistency_loss function computes the cycle consistency loss.
It passes the output frame and the source frame through the generator to reconstruct the source frame.
The L1 loss is calculated between the reconstructed source frame and the original source frame.


Contrastive Loss:

The contrastive_loss function computes the contrastive loss using cosine similarity.
It calculates the cosine similarity between positive pairs (output-source, output-driving) and negative pairs (output-random, source-random).
The loss is computed as the negative log likelihood of the positive pairs over the sum of positive and negative pair similarities.
The neg_pair_loss function calculates the loss for negative pairs using a margin.


Discriminator Loss:

The discriminator_loss function computes the loss for the discriminator.
It calculates the MSE loss between the predicted values for real samples and a tensor of ones, and the MSE loss between the predicted values for fake samples and a tensor of zeros.
The total discriminator loss is the sum of the real and fake losses.
'''
# Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.weights = weights
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg_face = models.vgg_face(pretrained=True).features
        self.gaze_loss = GazeLoss()
    
    def forward(self, output_frame, target_frame):
        loss_vgg = self.vgg_loss(output_frame, target_frame)
        loss_vgg_face = self.vgg_face_loss(output_frame, target_frame)
        loss_gaze = self.gaze_loss(output_frame, target_frame)
        
        total_loss = self.weights[0] * loss_vgg + self.weights[1] * loss_vgg_face + self.weights[2] * loss_gaze
        return total_loss
    
    def vgg_loss(self, output_frame, target_frame):
        output_features = self.vgg(output_frame)
        target_features = self.vgg(target_frame)
        return F.mse_loss(output_features, target_features)
    
    def vgg_face_loss(self, output_frame, target_frame):
        output_features = self.vgg_face(output_frame)
        target_features = self.vgg_face(target_frame)
        return F.mse_loss(output_features, target_features)

# Adversarial Loss
def adversarial_loss(output_frame, discriminator):
    fake_pred = discriminator(output_frame)
    loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    return loss

# Cycle Consistency Loss
def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss


# Create an instance of the PerceptualLoss class
perceptual_loss_fn = PerceptualLoss().to(device)

# Create an instance of the Encoder class
encoder = Encoder(input_nc=3, output_nc=256).to(device)

def contrastive_loss(output_frame, source_frame, driving_frame, margin=1.0, encoder=encoder):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    
    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]
    
    loss = 0
    for pos_pair in pos_pairs:
        loss += torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) / 
                          (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) + neg_pair_loss(pos_pair, neg_pairs)))
    
    return loss

def neg_pair_loss(pos_pair, neg_pairs):
    loss = 0
    for neg_pair in neg_pairs:
        loss += torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)
    return loss

# Discriminator Loss
def discriminator_loss(real_pred, fake_pred):
    real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
    fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    return real_loss + fake_loss


def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)
    
    for epoch in range(cfg.training.base_epochs):
        for batch in dataloader:
            video_frames = batch['images']
            video_id = batch['video_id']
            
            # Sample source and driving frames
            source_frame = video_frames[torch.randint(0, len(video_frames), (1,))]
            driving_frame = video_frames[torch.randint(0, len(video_frames), (1,))]
            
            # Prepare data
            source_frame = source_frame.to(device)
            driving_frame = driving_frame.to(device)
            
            # Train generator
            optimizer_G.zero_grad()
            
            # Generate output frame
            output_frame = Gbase(source_frame, driving_frame)
            
            # Compute losses
            loss_perceptual = perceptual_loss_fn(output_frame, driving_frame)
            loss_adversarial = adversarial_loss(output_frame, Dbase)
            loss_cycle_consistency = cycle_consistency_loss(output_frame, source_frame, driving_frame, Gbase)
            loss_contrastive = contrastive_loss(output_frame, source_frame, driving_frame)
            
            loss_G = loss_perceptual + loss_adversarial + loss_cycle_consistency + loss_contrastive
            
            # Backpropagate and update generator
            loss_G.backward()
            optimizer_G.step()
            
            # Train discriminator
            optimizer_D.zero_grad()
            
            # Compute discriminator loss
            real_pred = Dbase(driving_frame)
            fake_pred = Dbase(output_frame.detach())
            loss_D = discriminator_loss(real_pred, fake_pred)
            
            # Backpropagate and update discriminator
            loss_D.backward()
            optimizer_D.step()
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}")
        
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def train_hr(cfg, GHR, Dhr, dataloader_hr):
    GHR.train()
    Dhr.train()
    
    for epoch in range(cfg.training.hr_epochs):
        for batch in dataloader_hr:
            video_frames = batch['images']
            video_id = batch['video_id']
            
            # Load pre-trained base model Gbase
            # Freeze Gbase
            # Training loop for high-resolution model
            # ...

def train_student(cfg, Student, GHR, dataloader_avatars):
    Student.train()
    
    for epoch in range(cfg.training.student_epochs):
        for batch in dataloader_avatars:
            video_frames = batch['images']
            video_id = batch['video_id']
            
            # Load pre-trained high-resolution model GHR
            # Freeze GHR
            # Training loop for student model
            # ...

def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter()
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        n_sample_frames=cfg.training.n_sample_frames,
        sample_rate=cfg.training.sample_rate,
        img_scale=(1.0, 1.0),
        video_dir=cfg.training.video_dir,
        json_file=cfg.training.json_file,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
    Gbase = model.Gbase()
    Dbase = model.Discriminator()
    train_base(cfg, Gbase, Dbase, dataloader)
    
    GHR = model.GHR()
    GHR.Gbase.load_state_dict(Gbase.state_dict())
    Dhr = model.Discriminator()
    train_hr(cfg, GHR, Dhr, dataloader)
    
    Student = model.Student(num_avatars=100) # this should equal the number of celebs in dataset
    train_student(cfg, Student, GHR, dataloader)
    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(GHR.state_dict(), 'GHR.pth')
    torch.save(Student.state_dict(), 'Student.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1.yaml")
    main(config)