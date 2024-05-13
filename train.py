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
from model import MPGazeLoss,Encoder
from rome_losses import PerceptualLoss


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

def adversarial_loss(output_frame, discriminator):
    fake_pred = discriminator(output_frame)
    loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    return loss

def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss

def contrastive_loss(output_frame, source_frame, driving_frame, encoder,margin=1.0):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    z_rand = torch.randn_like(z_out)  # Define z_rand
    
    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]  # Update neg_pairs
    
    loss = 0
    for pos_pair in pos_pairs:
        loss += torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) / 
                          (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) + neg_pair_loss(pos_pair, neg_pairs, margin)))  # Pass margin to neg_pair_loss
    
    return loss

def neg_pair_loss(pos_pair, neg_pairs, margin):
    loss = 0
    for neg_pair in neg_pairs:
        loss += torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)  # Update margin
    return loss

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
    

    # Create an instance of the PerceptualLoss class
    perceptual_loss_fn = PerceptualLoss().to(device)
    gaze_loss_fn = MPGazeLoss(device)

    # Create an instance of the Encoder class
    encoder = Encoder(input_nc=3, output_nc=256).to(device)

    for epoch in range(cfg.training.base_epochs):
        for batch in dataloader:
            source_frames = batch['source_frames'].to(device)
            driving_frames = batch['driving_frames'].to(device)
            keypoints = batch['keypoints'].to(device)
            

            # Train generator
            optimizer_G.zero_grad()
            
            # Generate output frames
            output_frames = Gbase(source_frames, driving_frames)
            
            # Compute losses
            loss_perceptual = perceptual_loss_fn(output_frames, driving_frames)
            loss_adversarial = adversarial_loss(output_frames, Dbase)
            loss_cosine = contrastive_loss(output_frames, source_frames, driving_frames, encoder)
            loss_gaze = gaze_loss_fn(output_frames, driving_frames, keypoints)

            loss_G = cfg.training.lambda_perceptual * loss_perceptual + \
                    cfg.training.lambda_adversarial * loss_adversarial + \
                    cfg.training.lambda_cosine * loss_cosine + \
                    cfg.training.lambda_gaze * loss_gaze
                        
            
            # Backpropagate and update generator
            loss_G.backward()
            optimizer_G.step()
            
            # Train discriminator
            optimizer_D.zero_grad()
            
            # Compute discriminator loss
            real_pred = Dbase(driving_frames)
            fake_pred = Dbase(output_frames.detach())
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

def train_hr(cfg, GHR, Genh, dataloader_hr):
    GHR.train()
    Genh.train()
    


    # Create an instance of the PerceptualLoss class
    perceptual_loss_fn = PerceptualLoss().to(device)
    gaze_loss_fn = MPGazeLoss(device=device)

    optimizer_G = torch.optim.AdamW(Genh.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.hr_epochs, eta_min=1e-6)
    
    for epoch in range(cfg.training.hr_epochs):
        for batch in dataloader_hr:
            source_frames = batch['source_frames'].to(device)
            driving_frames = batch['driving_frames'].to(device)
            keypoints = batch['keypoints'].to(device)

            # Generate output frames using pre-trained base model
            with torch.no_grad():
                xhat_base = GHR.Gbase(source_frames, driving_frames)
            
            # Train high-resolution model
            optimizer_G.zero_grad()
            
            # Generate high-resolution output frames
            xhat_hr = Genh(xhat_base)
            
           
            # Compute losses
            loss_supervised = Genh.supervised_loss(xhat_hr, driving_frames)
            loss_unsupervised = Genh.unsupervised_loss(xhat_base, xhat_hr)
            loss_perceptual = perceptual_loss_fn(xhat_hr, driving_frames)
            loss_gaze = gaze_loss_fn(xhat_hr, driving_frames, keypoints)

            loss_G = cfg.training.lambda_supervised * loss_supervised + \
                    cfg.training.lambda_unsupervised * loss_unsupervised + \
                    cfg.training.lambda_perceptual * loss_perceptual + \
                    cfg.training.lambda_gaze * loss_gaze

            # loss_supervised = Genh.supervised_loss(xhat_hr, driving_frames)
            # loss_unsupervised = Genh.unsupervised_loss(xhat_base, xhat_hr)
            
            # loss_G = cfg.training.lambda_supervised * loss_supervised + cfg.training.lambda_unsupervised * loss_unsupervised
            
            # Backpropagate and update high-resolution model
            loss_G.backward()
            optimizer_G.step()
        
        # Update learning rate
        scheduler_G.step()
        
        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.hr_epochs}], "
                  f"Loss_G: {loss_G.item():.4f}")
        
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Genh.state_dict(), f"Genh_epoch{epoch+1}.pth")

def train_student(cfg, Student, GHR, dataloader_avatars):
    Student.train()
    
    optimizer_S = torch.optim.AdamW(Student.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    
    scheduler_S = CosineAnnealingLR(optimizer_S, T_max=cfg.training.student_epochs, eta_min=1e-6)
    
    for epoch in range(cfg.training.student_epochs):
        for batch in dataloader_avatars:
            avatar_indices = batch['avatar_indices'].to(device)
            driving_frames = batch['driving_frames'].to(device)
            
            # Generate high-resolution output frames using pre-trained HR model
            with torch.no_grad():
                xhat_hr = GHR(driving_frames)
            
            # Train student model
            optimizer_S.zero_grad()
            
            # Generate output frames using student model
            xhat_student = Student(driving_frames, avatar_indices)
            
            # Compute loss
            loss_S = F.mse_loss(xhat_student, xhat_hr)
            
            # Backpropagate and update student model
            loss_S.backward()
            optimizer_S.step()
        
        # Update learning rate
        scheduler_S.step()
        
        # Log and save checkpoints
        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.student_epochs}], "
                  f"Loss_S: {loss_S.item():.4f}")
        
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Student.state_dict(), f"Student_epoch{epoch+1}.pth")

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
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)