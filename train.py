import argparse
import torch
import model
import cv2 as cv
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
from rome_losses import Vgg19 # use vgg19 for perceptualloss 
import cv2
import mediapipe as mp
from memory_profiler import profile


face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)# this slows thing down - only for debug

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
# @profile
def adversarial_loss(output_frame, discriminator):
    fake_pred = discriminator(output_frame)
    loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    return loss.requires_grad_()

# @profile
def cycle_consistency_loss(output_frame, source_frame, driving_frame, generator):
    reconstructed_source = generator(output_frame, source_frame)
    loss = F.l1_loss(reconstructed_source, source_frame)
    return loss.requires_grad_()


def contrastive_loss(output_frame, source_frame, driving_frame, encoder, margin=1.0):
    z_out = encoder(output_frame)
    z_src = encoder(source_frame)
    z_drv = encoder(driving_frame)
    z_rand = torch.randn_like(z_out, requires_grad=True)

    pos_pairs = [(z_out, z_src), (z_out, z_drv)]
    neg_pairs = [(z_out, z_rand), (z_src, z_rand)]

    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for pos_pair in pos_pairs:
        loss = loss + torch.log(torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) /
                                (torch.exp(F.cosine_similarity(pos_pair[0], pos_pair[1])) +
                                 neg_pair_loss(pos_pair, neg_pairs, margin)))

    return loss

def neg_pair_loss(pos_pair, neg_pairs, margin):
    loss = torch.tensor(0.0, requires_grad=True).to(device)
    for neg_pair in neg_pairs:
        loss = loss + torch.exp(F.cosine_similarity(pos_pair[0], neg_pair[1]) - margin)
    return loss
# @profile
def discriminator_loss(real_pred, fake_pred):
    real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
    fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    return (real_loss + fake_loss).requires_grad_()


# @profile
def gaze_loss_fn(predicted_gaze, target_gaze, face_image):
    # Ensure face_image has shape (C, H, W)
    if face_image.dim() == 4 and face_image.shape[0] == 1:
        face_image = face_image.squeeze(0)
    if face_image.dim() != 3 or face_image.shape[0] not in [1, 3]:
        raise ValueError(f"Expected face_image of shape (C, H, W), got {face_image.shape}")
    
    # Convert face image from tensor to numpy array
    face_image = face_image.detach().cpu().numpy()
    if face_image.shape[0] == 3:  # if channels are first
        face_image = face_image.transpose(1, 2, 0)
    face_image = (face_image * 255).astype(np.uint8)

    # Extract eye landmarks using MediaPipe
    results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
    if not results.multi_face_landmarks:
        return torch.tensor(0.0, requires_grad=True).to(device)

    eye_landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
        right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
        eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))

    # Compute loss for each eye
    loss = 0.0
    h, w = face_image.shape[:2]
    for left_eye, right_eye in eye_landmarks:
        # Convert landmarks to pixel coordinates
        left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
        right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]

        # Create eye mask
        left_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        right_mask = torch.zeros((1, h, w), requires_grad=True).to(device)
        cv2.fillPoly(left_mask[0].cpu().numpy(), [np.array(left_eye_pixels)], 1.0)
        cv2.fillPoly(right_mask[0].cpu().numpy(), [np.array(right_eye_pixels)], 1.0)

        # Compute gaze loss for each eye
        left_gaze_loss = F.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
        right_gaze_loss = F.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
        loss += left_gaze_loss + right_gaze_loss

    return loss / len(eye_landmarks)


def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device)
    encoder = Encoder(input_nc=3, output_nc=256).to(device)

    for epoch in range(cfg.training.base_epochs):
        print("epoch:", epoch)
        for batch in dataloader:
            source_frames = batch['source_frames'] #.to(device)
            driving_frames = batch['driving_frames'] #.to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx].to(device)
                driving_frame = driving_frames[idx].to(device)

                # Train generator
                optimizer_G.zero_grad()
                output_frame = Gbase(source_frame, driving_frame)

                # Resize output_frame to 256x256 to match the driving_frame size
                resized_output_frame = F.interpolate(output_frame, size=(256, 256), mode='bilinear', align_corners=False)

                # Compute losses
                output_vgg_features = vgg19(resized_output_frame)
                driving_vgg_features = vgg19(driving_frame)
                loss_perceptual = 0

                for output_feat, driving_feat in zip(output_vgg_features, driving_vgg_features):
                    loss_perceptual = loss_perceptual + perceptual_loss_fn(output_feat, driving_feat.detach())


                loss_adversarial = adversarial_loss(output_frame, Dbase)
                #loss_cosine = contrastive_loss(output_frame, source_frame, driving_frame, encoder)
                loss_gaze = gaze_loss_fn(output_frame, driving_frame, source_frame)

                # Accumulate gradients
                loss_gaze.backward()
                loss_perceptual.backward(retain_graph=True)
                loss_adversarial.backward()
                # loss_cosine.backward(retain_graph=True)
                

                # Update generator
                optimizer_G.step()

                # Train discriminator
                optimizer_D.zero_grad()
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
                  f"Loss_G: {loss_gaze.item():.4f}, Loss_D: {loss_D.item():.4f}")
        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def train_hr(cfg, GHR, Genh, dataloader_hr):
    GHR.train()
    Genh.train()

    vgg19 = Vgg19().to(device)
    perceptual_loss_fn = nn.L1Loss().to(device)
    # gaze_loss_fn = MPGazeLoss(device=device)

    optimizer_G = torch.optim.AdamW(Genh.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.hr_epochs, eta_min=1e-6)

    for epoch in range(cfg.training.hr_epochs):
        for batch in dataloader_hr:
            source_frames = batch['source_frames'].to(device)
            driving_frames = batch['driving_frames'].to(device)

            num_frames = len(source_frames)  # Get the number of frames in the batch

            for idx in range(num_frames):
                source_frame = source_frames[idx]
                driving_frame = driving_frames[idx]

                # Generate output frame using pre-trained base model
                with torch.no_grad():
                    xhat_base = GHR.Gbase(source_frame, driving_frame)

                # Train high-resolution model
                optimizer_G.zero_grad()
                xhat_hr = Genh(xhat_base)


                # Compute losses - option 1
                # loss_supervised = Genh.supervised_loss(xhat_hr, driving_frame)
                # loss_unsupervised = Genh.unsupervised_loss(xhat_base, xhat_hr)
                # loss_perceptual = perceptual_loss_fn(xhat_hr, driving_frame)

                # option2 ? 🤷 use vgg19 as per metaportrait?
                # - Compute losses
                xhat_hr_vgg_features = vgg19(xhat_hr)
                driving_vgg_features = vgg19(driving_frame)
                loss_perceptual = 0
                for xhat_hr_feat, driving_feat in zip(xhat_hr_vgg_features, driving_vgg_features):
                    loss_perceptual += perceptual_loss_fn(xhat_hr_feat, driving_feat.detach())

                loss_supervised = perceptual_loss_fn(xhat_hr, driving_frame)
                loss_unsupervised = perceptual_loss_fn(xhat_hr, xhat_base)
                loss_gaze = gaze_loss_fn(xhat_hr, driving_frame)
                loss_G = (
                    cfg.training.lambda_supervised * loss_supervised
                    + cfg.training.lambda_unsupervised * loss_unsupervised
                    + cfg.training.lambda_perceptual * loss_perceptual
                    + cfg.training.lambda_gaze * loss_gaze
                )

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
    
    Gbase = model.Gbase().to(device)
    Dbase = model.Discriminator(input_nc=3).to(device) # 🤷
    
    train_base(cfg, Gbase, Dbase, dataloader)
    
    GHR = model.GHR()
    GHR.Gbase.load_state_dict(Gbase.state_dict())
    Dhr = model.Discriminator(input_nc=3).to(device) # 🤷
    train_hr(cfg, GHR, Dhr, dataloader)
    
    Student = model.Student(num_avatars=100) # this should equal the number of celebs in dataset
    train_student(cfg, Student, GHR, dataloader)
    
    torch.save(Gbase.state_dict(), 'Gbase.pth')
    torch.save(GHR.state_dict(), 'GHR.pth')
    torch.save(Student.state_dict(), 'Student.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)