import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet,Bottleneck, resnet18
from unet3d import ResBlock3D


import torch
import torch.nn as nn
import torchvision.models as models




class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3D_WS(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3D_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(
                                  dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, input_channels, output_channels):
        super().__init__()
        self.dimension = dimension
        self.input_channels = input_channels
        self.output_channels = output_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.input_channels, self.output_channels, 3, padding= 1)
            self.conv_ws = Conv2d_WS(in_channels = self.input_channels,
                                  out_channels= self.output_channels,
                                  kernel_size = 3,
                                  padding = 1)
            self.conv = nn.Conv2d(self.output_channels, self.output_channels, 3, padding = 1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
            self.conv_ws = Conv3D_WS(in_channels=self.input_channels,
                                     out_channels=self.output_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)


    def forward(self, x):
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        return output



class Eapp(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First part: producing volumetric features vs
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3)
        self.resblock_128 = ResBlock_Custom(dimension=2, input_channels=64, output_channels=128)
        self.resblock_256 = ResBlock_Custom(dimension=2, input_channels=128, output_channels=256)
        self.resblock_512 = ResBlock_Custom(dimension=2, input_channels=256, output_channels=512)
        self.resblock3D_96 = ResBlock_Custom(dimension=3, input_channels=96, output_channels=96)
        self.resblock3D_96_2 = ResBlock_Custom(dimension=3, input_channels=96, output_channels=96)
        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # Second part: producing global descriptor es
        self.resnet50 = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=512)

    def forward(self, x):
        # First part
        out = self.conv(x)
        out = self.resblock_128(out)
        out = self.avgpool(out)
        out = self.resblock_256(out)
        out = self.avgpool(out)
        out = self.resblock_512(out)
        out = self.avgpool(out)
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.conv_1(out)
        vs = out.view(out.size(0), 96, 16, out.size(2), out.size(3))
        vs = self.resblock3D_96(vs)
        vs = self.resblock3D_96_2(vs)

        # Second part
        es = self.resnet50(x)

        return vs, es


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)




class WarpGenerator(nn.Module):
    def __init__(self, input_channels):
        super(WarpGenerator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=2048, kernel_size=1, padding=0, stride=1)
        self.hidden_layer = nn.Sequential(
            ResBlock_Custom(dimension=3, input_channels=512, output_channels=256),
            nn.Upsample(scale_factor=(2, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=256, output_channels=128),
            nn.Upsample(scale_factor=(2, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=128, output_channels=64),
            nn.Upsample(scale_factor=(1, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=64, output_channels=32),
            nn.Upsample(scale_factor=(1, 2, 2)),
        )
        self.conv3D = nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        print("The output shape after first conv layer is " + str(out.size()))
        # reshape
        out = out.view(out.size(0), 512, 4, 16, 16)
        print("The output shape after reshaping is " + str(out.size()))

        out = self.hidden_layer(out)
        print("The output shape after hidden_layer is " + str(out.size()))
        out = F.group_norm(out, num_groups=32)
        print("The output shape after group_norm is " + str(out.size()))
        out = F.relu(out)
        print("The output shape after relu is " + str(out.size()))
        out = torch.tanh(out)
        print("The final output shape is : " + str(out.size()))

        return out




class G3d(nn.Module):
    def __init__(self, input_channels):
        super(G3d, self).__init__()
        self.downsampling = nn.Sequential(
            ResBlock3D(input_channels, 96),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(96, 192),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(192, 384),
            nn.AvgPool3d(kernel_size=2, stride=2),
            ResBlock3D(384, 768),
        )
        self.upsampling = nn.Sequential(
            ResBlock3D(768, 384),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(384, 192),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlock3D(192, 96),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        )
        self.final_conv = nn.Conv3d(96, 96, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.upsampling(x)
        x = self.final_conv(x)
        return x


class G2d(nn.Module):
    def __init__(self, input_channels):
        super(G2d, self).__init__()
        self.res_blocks = nn.Sequential(
            ResBlock2D(input_channels, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
        )
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.upsampling(x)
        return x


'''
In this expanded version of compute_rt_warp, we first compute the rotation matrix from the rotation parameters using the compute_rotation_matrix function. The rotation parameters are assumed to be a tensor of shape (batch_size, 3), representing rotation angles in degrees around the x, y, and z axes.
Inside compute_rotation_matrix, we convert the rotation angles from degrees to radians and compute the individual rotation matrices for each axis using the rotation angles. We then combine the rotation matrices using matrix multiplication to obtain the final rotation matrix.
Next, we create a 4x4 affine transformation matrix and set the top-left 3x3 submatrix to the computed rotation matrix. We also set the first three elements of the last column to the translation parameters.
Finally, we create a grid of normalized coordinates using F.affine_grid based on the affine transformation matrix. The grid size is assumed to be 64x64x64, but you can adjust it according to your specific requirements.
The resulting grid represents the warping transformations based on the given rotation and translation parameters, which can be used to warp the volumetric features or other tensors.
https://github.com/Kevinfringe/MegaPortrait/issues/4

'''
def compute_rt_warp(rotation, translation):
    # Compute the rotation matrix from the rotation parameters
    rotation_matrix = compute_rotation_matrix(rotation)

    # Create a 4x4 affine transformation matrix
    affine_matrix = torch.eye(4, device=rotation.device).repeat(rotation.shape[0], 1, 1)

    # Set the top-left 3x3 submatrix to the rotation matrix
    affine_matrix[:, :3, :3] = rotation_matrix

    # Set the first three elements of the last column to the translation parameters
    affine_matrix[:, :3, 3] = translation

    # Create a grid of normalized coordinates
    grid_size = 64  # Assumes a grid size of 64x64x64
    grid = F.affine_grid(affine_matrix[:, :3], (rotation.shape[0], 1, grid_size, grid_size, grid_size), align_corners=False)

    return grid

def compute_rotation_matrix(rotation):
    # Assumes rotation is a tensor of shape (batch_size, 3), representing rotation angles in degrees
    rotation_rad = rotation * (torch.pi / 180.0)  # Convert degrees to radians

    cos_alpha = torch.cos(rotation_rad[:, 0])
    sin_alpha = torch.sin(rotation_rad[:, 0])
    cos_beta = torch.cos(rotation_rad[:, 1])
    sin_beta = torch.sin(rotation_rad[:, 1])
    cos_gamma = torch.cos(rotation_rad[:, 2])
    sin_gamma = torch.sin(rotation_rad[:, 2])

    # Compute the rotation matrix using the rotation angles
    zero = torch.zeros_like(cos_alpha)
    one = torch.ones_like(cos_alpha)

    R_alpha = torch.stack([
        torch.stack([one, zero, zero], dim=1),
        torch.stack([zero, cos_alpha, -sin_alpha], dim=1),
        torch.stack([zero, sin_alpha, cos_alpha], dim=1)
    ], dim=1)

    R_beta = torch.stack([
        torch.stack([cos_beta, zero, sin_beta], dim=1),
        torch.stack([zero, one, zero], dim=1),
        torch.stack([-sin_beta, zero, cos_beta], dim=1)
    ], dim=1)

    R_gamma = torch.stack([
        torch.stack([cos_gamma, -sin_gamma, zero], dim=1),
        torch.stack([sin_gamma, cos_gamma, zero], dim=1),
        torch.stack([zero, zero, one], dim=1)
    ], dim=1)

    # Combine the rotation matrices
    rotation_matrix = torch.matmul(R_alpha, torch.matmul(R_beta, R_gamma))

    return rotation_matrix


class WarpingGenerator(nn.Module):
    def __init__(self, input_channels):
        super(WarpingGenerator, self).__init__()
        self.emotion_net = nn.Sequential(
            ResBlock3D_Adaptive(input_channels, 256),
            ResBlock3D_Adaptive(256, 128),
            ResBlock3D_Adaptive(128, 64),
            nn.Conv3d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, rotation, translation, emotion, appearance):
        # Compute rotation and translation warp
        rt_warp = compute_rt_warp(rotation, translation)
        
        # Compute emotion warp
        emotion_warp = self.emotion_net(emotion + appearance)
        
        # Combine warps
        warp = rt_warp + emotion_warp
        
        return warp
        

'''

In this expanded version of AdaptiveGroupNorm, we define a module that performs adaptive group normalization on a 5D input tensor of shape (batch_size, channels, depth, height, width).
The module is initialized with the number of channels (num_channels), the number of groups (num_groups), and a small constant (eps) for numerical stability.
We define learnable parameters gamma and beta for adaptive normalization, which have the same shape as the input tensor.
We also define embedding layers embed_gamma and embed_beta that take the mean of the input tensor across the spatial dimensions (depth, height, width) and output adaptive parameters of the same shape as gamma and beta. These embedding layers allow the normalization parameters to adapt based on the input.
In the forward method, we first reshape the input tensor into (batch_size, num_groups, channels // num_groups, depth, height, width) to perform group normalization. We compute the mean and variance per group and normalize the tensor using these statistics.
After normalization, we reshape the tensor back to its original shape (batch_size, channels, depth, height, width).
Finally, we compute the adaptive scale and shift parameters by adding the learnable parameters gamma and beta to the embedded parameters obtained from embed_gamma and embed_beta. We apply these adaptive parameters to the normalized tensor using element-wise multiplication and addition.
The resulting tensor has the same shape as the input tensor but has undergone adaptive group normalization.
This AdaptiveGroupNorm module can be used as a replacement for standard group normalization layers in the network, allowing for adaptive normalization based on the input.
'''
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super(AdaptiveGroupNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps

        # Parameters for adaptive normalization
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

        # Embedding layers for adaptive parameters
        self.embed_gamma = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, num_channels)
        )
        self.embed_beta = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        # Reshape into (batch_size, num_groups, channels // num_groups, depth, height, width)
        x = x.view(batch_size, self.num_groups, -1, depth, height, width)

        # Compute mean and variance per group
        mean = x.mean(dim=[2, 3, 4, 5], keepdim=True)
        var = x.var(dim=[2, 3, 4, 5], keepdim=True, unbiased=False)

        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)

        # Reshape back to (batch_size, channels, depth, height, width)
        x = x.view(batch_size, channels, depth, height, width)

        # Adaptive scale and shift
        embedded_gamma = self.embed_gamma(x.mean(dim=[2, 3, 4], keepdim=True)).view(batch_size, channels, 1, 1, 1)
        embedded_beta = self.embed_beta(x.mean(dim=[2, 3, 4], keepdim=True)).view(batch_size, channels, 1, 1, 1)
        adaptive_gamma = self.gamma + embedded_gamma
        adaptive_beta = self.beta + embedded_beta

        return x * adaptive_gamma + adaptive_beta


class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(output_channels)
        self.norm2 = AdaptiveGroupNorm(output_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = F.relu(out)
        return out

class Emtn(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_pose_net = resnet18(pretrained=True)
        self.head_pose_net.fc = nn.Linear(512, 6)  # 6 corresponds to rotation and translation parameters
        
        self.expression_net = resnet18(num_classes=50)  # 50 corresponds to the dimensions of expression vector

    def forward(self, x):
        head_pose = self.head_pose_net(x)
        expression = self.expression_net(x)
        return head_pose, expression

'''
The main changes made to align the code with the training stages are:

Introduced Gbase class that combines the components of the base model.
Introduced Genh class for the high-resolution model.
Introduced GHR class that combines the base model Gbase and the high-resolution model Genh.
Introduced Student class for the student model, which includes an encoder, decoder, and SPADE blocks for avatar conditioning.
Added separate training functions for each stage: train_base, train_hr, and train_student.
Demonstrated the usage of the training functions and saving the trained models.

Note: The code assumes the presence of appropriate dataloaders (dataloader, dataloader_hr, dataloader_avatars) and the implementation of the SPADEResBlock class for the student model. Additionally, the specific training loop details and loss functions need to be implemented based on the paper's description.

The main changes made to align the code with the paper are:

The Emtn (motion encoder) is now a single module that outputs the rotation parameters (Rs, Rd), translation parameters (ts, td), and expression vectors (zs, zd) for both the source and driving images.
The warping generators (Ws2c and Wc2d) now take the rotation, translation, expression, and appearance features as separate inputs, as described in the paper.
The warping process is updated to match the paper's description. First, the volumetric features (vs) are warped using ws2c to obtain the canonical volume (vc). Then, vc is processed by G3d to obtain vc2d. Finally, vc2d is warped using wc2d to impose the driving motion.
The orthographic projection (denoted as P in the paper) is implemented as an average pooling operation followed by a squeeze operation to reduce the spatial dimensions.
The projected features (vc2d_projected) are passed through G2d to obtain the final output image (xhat).

These changes align the code more closely with the architecture and processing steps described in the MegaPortraits paper.
'''

class Gbase(nn.Module):
    def __init__(self):
        super(Gbase, self).__init__()
        self.Eapp = Eapp()
        self.Emtn = Emtn()
        self.Ws2c = WarpingGenerator(input_channels=512)
        self.Wc2d = WarpingGenerator(input_channels=512)
        self.G3d = G3d(input_channels=96)
        self.G2d = G2d(input_channels=96)

    def forward(self, xs, xd):
        vs, es = self.Eapp(xs)
        Rs, ts, zs = self.Emtn(xs)
        Rd, td, zd = self.Emtn(xd)
        
        ws2c = self.Ws2c(Rs, ts, zs, es)
        wc2d = self.Wc2d(Rd, td, zd, es)
        
        vc = torch.nn.functional.grid_sample(vs, ws2c)
        vc2d = self.G3d(vc)
        vc2d = torch.nn.functional.grid_sample(vc2d, wc2d)
        
        vc2d_projected = torch.nn.functional.avg_pool3d(vc2d, kernel_size=(1, 1, vc2d.size(4))).squeeze(4)
        xhat = self.G2d(vc2d_projected)
        
        return xhat


# original resblock
class ResBlock2D(nn.Module):
    def __init__(self, n_c, kernel=3, dilation=1, p_drop=0.15):
        super(ResBlock2D, self).__init__()
        padding = self._get_same_padding(kernel, dilation)

        layer_s = list()
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, padding=padding, dilation=dilation, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=True))
        # dropout
        layer_s.append(nn.Dropout(p_drop))
        # convolution
        layer_s.append(nn.Conv2d(n_c, n_c, kernel, dilation=dilation, padding=padding, bias=False))
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        self.layer = nn.Sequential(*layer_s)
        self.final_activation = nn.ELU(inplace=True)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return self.final_activation(x + out)


class Genh(nn.Module):
    def __init__(self):
        super(Genh, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            ResBlock2D(64, 128),
            ResBlock2D(128, 256),
            ResBlock2D(256, 512),
        )
        self.decoder = nn.Sequential(
            ResBlock2D(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def unsupervised_loss(self, x, x_hat):
        # Cycle consistency loss
        x_cycle = self.forward(x_hat)
        cycle_loss = F.l1_loss(x_cycle, x)

        # Other unsupervised losses can be added here

        return cycle_loss

    def supervised_loss(self, x_hat, y):
        # Supervised losses
        l1_loss = F.l1_loss(x_hat, y)
        perceptual_loss = self.perceptual_loss(x_hat, y)

        return l1_loss + perceptual_loss

    def perceptual_loss(self, x, y):
        # Implement perceptual loss using a pre-trained VGG network
        pass

class GHR(nn.Module):
    def __init__(self):
        super(GHR, self).__init__()
        self.Gbase = Gbase()
        self.Genh = Genh()

    def forward(self, xs, xd):
        xhat_base = self.Gbase(xs, xd)
        xhat_hr = self.Genh(xhat_base)
        return xhat_hr

class Student(nn.Module):
    def __init__(self, num_avatars):
        super(Student, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            ResBlock_Custom(dimension=2, input_channels=64, output_channels=128),
            ResBlock_Custom(dimension=2, input_channels=128, output_channels=256),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=512),
            ResBlock_Custom(dimension=2, input_channels=512, output_channels=1024)
        )
        self.decoder = nn.Sequential(
            ResBlock_Custom(dimension=2, input_channels=1024, output_channels=512),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=512, output_channels=256),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=128),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        )
        self.spade_blocks = nn.ModuleList([
            SPADEResBlock(1024, 1024, num_avatars) for _ in range(4)
        ])

    def forward(self, xd, avatar_index):
        encoded = self.encoder(xd)
        for spade_block in self.spade_blocks:
            encoded = spade_block(encoded, avatar_index)
        xhat = self.decoder(encoded)
        return xhat


'''
In this expanded code, we have the SPADEResBlock class which represents a residual block with SPADE (Spatially-Adaptive Normalization) layers. The block consists of two convolutional layers (conv_0 and conv_1) with normalization layers (norm_0 and norm_1) and a learnable shortcut connection (conv_s and norm_s) if the input and output channels differ.
The SPADE class implements the SPADE layer, which learns to modulate the normalized activations based on the avatar embedding. It consists of a shared convolutional layer (conv_shared) followed by separate convolutional layers for gamma and beta (conv_gamma and conv_beta). The avatar embeddings (avatar_shared_emb, avatar_gamma_emb, and avatar_beta_emb) are learned for each avatar index and are added to the corresponding activations.
During the forward pass of SPADEResBlock, the input x is passed through the shortcut connection and the main branch. The main branch applies the SPADE normalization followed by the convolutional layers. The output of the block is the sum of the shortcut and the main branch activations.
The SPADE layer first normalizes the input x using instance normalization. It then applies the shared convolutional layer to obtain the shared embedding. The gamma and beta values are computed by adding the avatar embeddings to the shared embedding and passing them through the respective convolutional layers. Finally, the normalized activations are modulated using the computed gamma and beta values.
Note that this implementation assumes the presence of the avatar index tensor avatar_index during the forward pass, which is used to retrieve the corresponding avatar embeddings.
'''
class SPADEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_avatars):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        middle_channels = min(in_channels, out_channels)

        self.conv_0 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.norm_0 = SPADE(in_channels, num_avatars)
        self.norm_1 = SPADE(middle_channels, num_avatars)

        if self.learned_shortcut:
            self.norm_s = SPADE(in_channels, num_avatars)

    def forward(self, x, avatar_index):
        x_s = self.shortcut(x, avatar_index)

        dx = self.conv_0(self.actvn(self.norm_0(x, avatar_index)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, avatar_index)))

        out = x_s + dx

        return out

    def shortcut(self, x, avatar_index):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, avatar_index))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, num_avatars):
        super().__init__()
        self.num_avatars = num_avatars
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.conv_shared = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

        self.avatar_shared_emb = nn.Embedding(num_avatars, 128)
        self.avatar_gamma_emb = nn.Embedding(num_avatars, norm_nc)
        self.avatar_beta_emb = nn.Embedding(num_avatars, norm_nc)

    def forward(self, x, avatar_index):
        avatar_shared = self.avatar_shared_emb(avatar_index)
        avatar_gamma = self.avatar_gamma_emb(avatar_index)
        avatar_beta = self.avatar_beta_emb(avatar_index)

        x = self.norm(x)
        shared_emb = self.conv_shared(x)
        gamma = self.conv_gamma(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        beta = self.conv_beta(shared_emb + avatar_shared.view(-1, 128, 1, 1))
        gamma = gamma + avatar_gamma.view(-1, self.norm_nc, 1, 1)
        beta = beta + avatar_beta.view(-1, self.norm_nc, 1, 1)

        out = x * (1 + gamma) + beta
        return out


'''

Gaze Loss:

The GazeLoss class computes the gaze loss between the output and target frames.
It uses a pre-trained GazeModel to predict the gaze directions for both frames.
The MSE loss is calculated between the predicted gaze directions.


Gaze Model:

The GazeModel class is based on the VGG16 architecture.
It uses the pre-trained VGG16 model as the base and modifies the classifier to predict gaze directions.
The classifier is modified to remove the last layer and add a fully connected layer that outputs a 2-dimensional gaze vector.


Encoder Model:

The Encoder class is a convolutional neural network that encodes the input frames into a lower-dimensional representation.
It consists of a series of convolutional layers with increasing number of filters, followed by batch normalization and ReLU activation.
The encoder performs downsampling at each stage to reduce the spatial dimensions.
The final output is obtained by applying adaptive average pooling and a 1x1 convolutional layer to produce the desired output dimensionality.



The GazeLoss can be used in the PerceptualLoss class to compute the gaze loss component. The Encoder model can be used in the contrastive_loss function to encode the frames into lower-dimensional representations for computing the cosine similarity.
'''
# Gaze Loss
class GazeLoss(nn.Module):
    def __init__(self):
        super(GazeLoss, self).__init__()
        self.gaze_model = GazeModel()
        self.loss_fn = nn.MSELoss()
    
    def forward(self, output_frame, target_frame):
        output_gaze = self.gaze_model(output_frame)
        target_gaze = self.gaze_model(target_frame)
        loss = self.loss_fn(output_gaze, target_gaze)
        return loss

# Gaze Model
class GazeModel(nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-1])
        self.gaze_fc = nn.Linear(4096, 2)
    
    def forward(self, x):
        features = self.base_model(x)
        gaze = self.gaze_fc(features)
        return gaze

# Encoder Model
class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        model += [nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Conv2d(ngf * (2 ** n_downsampling), output_nc, kernel_size=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)