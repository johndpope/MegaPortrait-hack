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
from omegaconf import OmegaConf
from torchvision import models
from model import PatchGanEncoder, PerceptualLoss, crop_and_warp_face, get_foreground_mask,remove_background_and_convert_to_rgb,apply_warping_field
import mediapipe as mp
import torchvision.transforms as transforms
import os
import torchvision.utils as vutils
import time
from torch.cuda.amp import autocast, GradScaler


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# In the adversarial_loss function, we now use the hinge loss for the generator.
#  The loss is calculated as the negative mean of the discriminator's prediction 
# for the fake frame. This encourages the generator to produce frames that can fool 
# the discriminator.
def adversarial_loss(output_frame, discriminator):
    fake_pred, fake_features  = discriminator(output_frame)
    loss = -torch.mean(fake_pred)
    return loss.requires_grad_()


def discriminator_loss(real_pred, fake_pred):
    real_loss = torch.mean(torch.relu(1 - real_pred))
    fake_loss = torch.mean(torch.relu(1 + fake_pred))
    return (real_loss + fake_loss).requires_grad_()
    
def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += torch.mean(torch.abs(real_feat - fake_feat))
    return loss

def cycle_consistency_loss(model,   Rs, ts, zs, zd,  Rd, td, vs, w_s2c, es):  
    vc = apply_warping_field(vs, w_s2c)
    vc3d = model.G3d(vc)
    w_c2d = model.warp_generator_c2d(Rd, td, zd, es)
    vc3d_warped = apply_warping_field(vc3d, w_c2d)
    vc2d_projected = torch.sum(vc3d_warped, dim=2)
    xhat_cycle = model.G2d(vc2d_projected)
    loss = F.l1_loss(xhat_cycle, model.G2d(torch.sum(vc, dim=2)))
    return loss





# a novel contrastive loss that allows our system to achieve higher degrees of disentanglement 
# between the latent motion and appearance representation
# Following the previous works, we train a multi-scale patch discriminator [ 42 ] Patchgan = encoder
#  with a hinge adversarial loss alongside the generator Gbase
def contrastive_loss_patchgan(output_frame, source_frame, driving_frame, encoder, margin=1.0):
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

# cosine distance formula
# s · (⟨zi, zj⟩ − m)
def contrastive_loss_1(pos_pairs, neg_pairs, margin=1.0, s=1.0):
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


def train_base(cfg, Gbase, Dbase, dataloader):
    Gbase.train()
    Dbase.train()
    optimizer_G = torch.optim.AdamW(Gbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    optimizer_D = torch.optim.AdamW(Dbase.parameters(), lr=cfg.training.lr, betas=(0.5, 0.999), weight_decay=1e-2)
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=cfg.training.base_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=cfg.training.base_epochs, eta_min=1e-6)

    perceptual_loss_fn = PerceptualLoss(device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0})
    encoder = PatchGanEncoder(input_nc=3, output_nc=256).to(device)
    scaler = GradScaler()

    for epoch in range(cfg.training.base_epochs):
        print("Epoch:", epoch)
        

        for batch in dataloader:

                source_frames = batch['source_frames']
                driving_frames = batch['driving_frames']
                video_id = batch['video_id'][0]

                # Access videos from dataloader2 for cycle consistency
                source_frames2 = batch['source_frames_star']
                driving_frames2 = batch['driving_frames_star']
                video_id2 = batch['video_id_star'][0]


                num_frames = len(driving_frames)
                len_source_frames = len(source_frames)
                len_driving_frames = len(driving_frames)
                len_source_frames2 = len(source_frames2)
                len_driving_frames2 = len(driving_frames2)

                for idx in range(num_frames):
                    # loop around if idx exceeds video length
                    source_frame = source_frames[idx % len_source_frames].to(device)
                    driving_frame = driving_frames[idx % len_driving_frames].to(device)

                    source_frame_star = source_frames2[idx % len_source_frames2].to(device)
                    driving_frame_star = driving_frames2[idx % len_driving_frames2].to(device)


                    with autocast():

                        # We use multiple loss functions for training, which can be split  into two groups.
                        # The first group consists of the standard training objectives for image synthesis. 
                        # These include perceptual [14] and GAN [ 33 ] losses that match 
                        # the predicted image ˆx𝑠→𝑑 to the  ground-truth x𝑑 . 
                        output_frame = Gbase(source_frame, driving_frame)

                        # Obtain the foreground mask for the driving image
                        foreground_mask = get_foreground_mask(source_frame)

                        # Move the foreground mask to the same device as output_frame
                        foreground_mask = foreground_mask.to(output_frame.device)

                        # Multiply the predicted and driving images with the foreground mask
                        masked_predicted_image = output_frame * foreground_mask
                        masked_target_image = driving_frame * foreground_mask

                        save_images = True
                        # Save the images
                        if save_images:
                            vutils.save_image(source_frame, f"{output_dir}/source_frame_{idx}.png")
                            vutils.save_image(driving_frame, f"{output_dir}/driving_frame_{idx}.png")
                            vutils.save_image(output_frame, f"{output_dir}/output_frame_{idx}.png")
                            vutils.save_image(source_frame_star, f"{output_dir}/source_frame_star_{idx}.png")
                            vutils.save_image(driving_frame_star, f"{output_dir}/driving_frame_star_{idx}.png")
                            vutils.save_image(masked_predicted_image, f"{output_dir}/masked_predicted_image_{idx}.png")
                            vutils.save_image(masked_target_image, f"{output_dir}/masked_target_image_{idx}.png")

                        # Calculate perceptual losses
                        loss_G_per = perceptual_loss_fn(masked_predicted_image, masked_target_image)
                        # Calculate adversarial losses
                        loss_G_adv = adversarial_loss(masked_predicted_image, Dbase)
                        loss_fm = perceptual_loss_fn(masked_predicted_image, masked_target_image, use_fm_loss=True)
                    
                    
                        
                        # The other objective CycleGAN regularizes the training and introduces disentanglement between the motion and canonical space
                        # In order to calculate this loss, we use an additional source-driving  pair x𝑠∗ and x𝑑∗ , 
                        # which is sampled from a different video! and therefore has different appearance from the current x𝑠 , x𝑑 pair.

                        # produce the following cross-reenacted image: ˆx𝑠∗→𝑑 = Gbase (x𝑠∗ , x𝑑 )
                        cross_reenacted_image = Gbase(source_frame_star, driving_frame)
                        if save_images:
                            vutils.save_image(cross_reenacted_image, f"{output_dir}/cross_reenacted_image_{idx}.png")

                        # Store the motion descriptors z𝑠→𝑑(predicted) and z𝑠∗→𝑑 (star predicted) from the 
                        # respective forward passes of the base network.
                        _, _, z_pred = Gbase.motionEncoder(output_frame) 
                        _, _, zd = Gbase.motionEncoder(driving_frame) 
                        
                        _, _, z_star__pred = Gbase.motionEncoder(cross_reenacted_image) 
                        _, _, zd_star = Gbase.motionEncoder(driving_frame_star) 

              
                        # Calculate cycle consistency loss 
                        # We then arrange the motion descriptors into positive pairs P that
                        # should align with each other: P = (z𝑠→𝑑 , z𝑑 ), (z𝑠∗→𝑑 , z𝑑 ) , and
                        # the negative pairs: N = (z𝑠→𝑑 , z𝑑∗ ), (z𝑠∗→𝑑 , z𝑑∗ ) . These pairs are
                        # used to calculate the following cosine distance:

                        P = [(z_pred, zd)     ,(z_star__pred, zd)]
                        N = [(z_pred, zd_star),(z_star__pred, zd_star)]
                        loss_G_cos = contrastive_loss_1(P, N)

                        loss_G_cos = contrastive_loss_patchgan(cross_reenacted_image, source_frame_star, driving_frame, encoder)

                        # loss_G_cyc = cycle_consistency_loss(Gbase,  Rs, ts, zs, zd,  Rd, td, vs, w_s2c, es) #  🤷 is this true? there's no weights indicated for this only cos
                    

                        # Combine the losses
                        total_loss = cfg.training.w_per * loss_G_per + \
                            cfg.training.w_adv * loss_G_adv + \
                            cfg.training.w_fm * loss_fm + \
                            cfg.training.w_cos * loss_G_cos 
                            # cfg.training.w_cyc * loss_G_cyc # added on - 
                        

                    
                        # Backpropagate and update generator
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer_G)
                        scaler.update()


                        # Train discriminator
                        optimizer_D.zero_grad()


                        with autocast():
                            # Calculate adversarial losses
                            real_pred, real_features = Dbase(driving_frame)
                            fake_pred, fake_features = Dbase(output_frame.detach())
                            loss_G_adv = adversarial_loss(output_frame, Dbase)
                            loss_D = discriminator_loss(real_pred, fake_pred)
                            loss_fm = feature_matching_loss(real_features, fake_features)


                    scaler.scale(loss_D).backward()
                    scaler.step(optimizer_D)
                    scaler.update()
                    optimizer_D.zero_grad()

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % cfg.training.log_interval == 0:
            print(f"Epoch [{epoch+1}/{cfg.training.base_epochs}], "
                  f"Loss_G: {loss_G_cos.item():.4f}, Loss_D: {loss_D.item():.4f}")

        if (epoch + 1) % cfg.training.save_interval == 0:
            torch.save(Gbase.state_dict(), f"Gbase_epoch{epoch+1}.pth")
            torch.save(Dbase.state_dict(), f"Dbase_epoch{epoch+1}.pth")

def unnormalize(tensor):
    """
    Unnormalize a tensor using the specified mean and std.
    
    Args:
    tensor (torch.Tensor): The normalized tensor.
    mean (list): The mean used for normalization.
    std (list): The std used for normalization.
    
    Returns:
    torch.Tensor: The unnormalized tensor.
    """
    # Check if the tensor is on a GPU and if so, move it to the CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Ensure tensor is a float and detach it from the computation graph
    tensor = tensor.float().detach()
    
    # Unnormalize
    # Define mean and std used for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    return tensor


def main(cfg: OmegaConf) -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = EMODataset(
        use_gpu=use_cuda,
        remove_background=True,
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
    Dbase = model.Discriminator(input_nc=3).to(device)
    
    train_base(cfg, Gbase, Dbase, dataloader)    
    torch.save(Gbase.state_dict(), 'Gbase.pth')

if __name__ == "__main__":
    config = OmegaConf.load("./configs/training/stage1-base.yaml")
    main(config)