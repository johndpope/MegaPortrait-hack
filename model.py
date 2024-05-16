import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet,Bottleneck, resnet18
import torchvision.models as models
import math
# from samsunglabs / rome
from src.networks.common.norm_layers import AdaptiveGroupNorm
from src.networks.common.conv_layers import WSConv2d



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



class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, input_channels, output_channels):
        super().__init__()
        self.dimension = dimension
        self.input_channels = input_channels
        self.output_channels = output_channels
        if dimension == 2:
            self.conv1 = nn.Conv2d(self.input_channels, self.output_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1)
        elif dimension == 3:
            self.conv1 = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
            self.conv2 = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.output_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.output_channels)

        if self.input_channels != self.output_channels:
            if dimension == 2:
                self.shortcut = nn.Conv2d(self.input_channels, self.output_channels, 1)
            elif dimension == 3:
                self.shortcut = nn.Conv3d(self.input_channels, self.output_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = nn.ReLU(inplace=True)(out)
        return out


'''
Eapp Class:

The Eapp class represents the appearance encoder (Eapp) in the diagram.
It consists of two parts: producing volumetric features (vs) and producing a global descriptor (es).

Producing Volumetric Features (vs):

The conv layer corresponds to the 7x7-Conv-64 block in the diagram.
The resblock_128, resblock_256, resblock_512 layers correspond to the ResBlock2D-128, ResBlock2D-256, ResBlock2D-512 blocks respectively, with average pooling (self.avgpool) in between.
The conv_1 layer corresponds to the GN, ReLU, 1x1-Conv2D-1536 block in the diagram.
The output of conv_1 is reshaped to (batch_size, 96, 16, height, width) and passed through resblock3D_96 and resblock3D_96_2, which correspond to the two ResBlock3D-96 blocks in the diagram.
The final output of this part is the volumetric features (vs).

Producing Global Descriptor (es):

The resnet50 layer corresponds to the ResNet50 block in the diagram.
It takes the input image (x) and produces the global descriptor (es).

Forward Pass:

During the forward pass, the input image (x) is passed through both parts of the Eapp network.
The first part produces the volumetric features (vs) by passing the input through the convolutional layers, residual blocks, and reshaping operations.
The second part produces the global descriptor (es) by passing the input through the ResNet50 network.
The Eapp network returns both vs and es as output.

In summary, the Eapp class in the code aligns well with the appearance encoder (Eapp) shown in the diagram. The network architecture follows the same structure, with the corresponding layers and blocks mapped accurately. The conv, resblock_128, resblock_256, resblock_512, conv_1, resblock3D_96, and resblock3D_96_2 layers in the code correspond to the respective blocks in the diagram for producing volumetric features. The resnet50 layer in the code corresponds to the ResNet50 block in the diagram for producing the global descriptor.
'''
class Eapp(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First part: producing volumetric features vs
        self.conv = WSConv2d(3, 64, 7, stride=1, padding=3)
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
        assert out.shape[1] == 64, f"Expected 64 channels after conv, got {out.shape[1]}"
        
        out = self.resblock_128(out)
        assert out.shape[1] == 128, f"Expected 128 channels after resblock_128, got {out.shape[1]}"
        out = self.avgpool(out)
        
        out = self.resblock_256(out)
        assert out.shape[1] == 256, f"Expected 256 channels after resblock_256, got {out.shape[1]}"
        out = self.avgpool(out)
        
        out = self.resblock_512(out)
        assert out.shape[1] == 512, f"Expected 512 channels after resblock_512, got {out.shape[1]}"
        out = self.avgpool(out)
        
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.conv_1(out)
        assert out.shape[1] == 1536, f"Expected 1536 channels after conv_1, got {out.shape[1]}"
        
        vs = out.view(out.size(0), 96, 16, out.size(2), out.size(3))
        assert vs.shape[1:] == (96, 16, out.size(2), out.size(3)), f"Expected vs shape (_, 96, 16, _, _), got {vs.shape}"
        
        vs = self.resblock3D_96(vs)
        assert vs.shape[1] == 96, f"Expected 96 channels after resblock3D_96, got {vs.shape[1]}"
        vs = self.resblock3D_96_2(vs) 
        assert vs.shape[1] == 96, f"Expected 96 channels after resblock3D_96_2, got {vs.shape[1]}"

        # Second part
        es = self.resnet50(x)
        assert es.shape[1] == 512, f"Expected 512 channels for es, got {es.shape[1]}"

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


# W→d |Ws→
'''

WarpGenerator Class:

The WarpGenerator class represents the warping generator (Ws2c or Wc2d) in the diagram.
It takes rotation, translation, expression, and appearance features as input and generates the warping field.

Input Processing:

The forward method takes rotation, translation, expression, and appearance features as separate inputs.
These inputs are concatenated along the channel dimension using torch.cat to obtain the combined input x.

1x1 Convolution:

The conv1 layer corresponds to the 1x1-Conv-2048 block in the diagram.
It takes the concatenated input x and applies a 1x1 convolution to reduce the number of channels to 2048.

Reshaping:

The output of conv1 is reshaped to (batch_size, 512, 4, 16, 16) using out.view(...).
This reshaping operation corresponds to the Reshape C2048 → C512xD4 block in the diagram.

Hidden Layers:

The hidden_layer consists of a series of ResBlock3D_Adaptive and upsampling operations.
The architecture of the hidden layer follows the structure shown in the diagram:

ResBlock3D_Adaptive(512, 256) corresponds to the ResBlock3D*-256, Upsample(2,2,2) block.
ResBlock3D_Adaptive(256, 128) corresponds to the ResBlock3D*-128, Upsample(2,2,2) block.
ResBlock3D_Adaptive(128, 64) corresponds to the ResBlock3D*-64, Upsample(1,2,2) block.
ResBlock3D_Adaptive(64, 32) corresponds to the ResBlock3D*-32, Upsample(1,2,2) block.



Output Convolution:

The conv3D layer corresponds to the Conv3D-3 block in the diagram.
It takes the output of the hidden layers and applies a 3D convolution with a kernel size of 3 and padding of 1 to produce the final warping field.

Activation Functions:

The output of the hidden layers is passed through group normalization (F.group_norm) and ReLU activation (F.relu).
The final output of the WarpGenerator is passed through a tanh activation function (torch.tanh) to obtain the warping field in the range [-1, 1].

In summary, the WarpGenerator class in the code aligns well with the warping generator (Ws2c or Wc2d) shown in the diagram. The input processing, 1x1 convolution, reshaping, hidden layers, and output convolution in the code correspond to the respective blocks in the diagram. The activation functions and upsampling operations are also consistent with the diagram.


This code first applies the 1D convolution (self.conv1) to the input tensor x, which reduces the number of channels to 2048. 
Then, the out.view(out.size(0), 512, 4, 16, 16) line reshapes the output tensor to have dimensions (batch_size, 512, 4, 16, 16), effectively transforming the shape from C2048 to C512xD4.
The dimensions in the out.view function call correspond to:

out.size(0): The batch size dimension
512: The number of channels (C) in the reshaped tensor
4: The depth dimension (D) in the reshaped tensor
16: The height dimension in the reshaped tensor
16: The width dimension in the reshaped tensor

This reshaping operation aligns with the "Reshape C2048 → C512xD4" block shown in the diagram, which takes the output of the 1x1 convolution and reshapes it into the specified dimensions.
        
'''


class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3D_Adaptive, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adaptive_norm1 = AdaptiveGroupNorm(32, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adaptive_norm2 = AdaptiveGroupNorm(32, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None
        
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.adaptive_norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.adaptive_norm2(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        out += residual
        out = self.relu2(out)
        out = self.upsample(out)
        
        return out

class WarpGenerator(nn.Module):
    def __init__(self):
        super(WarpGenerator, self).__init__()
        self.conv_initial = nn.Conv3d(1, 2048, kernel_size=1)
        self.res_blocks = nn.Sequential(
            ResBlock3D_Adaptive(2048, 256),
            ResBlock3D_Adaptive(256, 128),
            ResBlock3D_Adaptive(128, 64),
            ResBlock3D_Adaptive(64, 32)
        )
        self.final_layers = nn.Sequential(
            AdaptiveGroupNorm(32, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, Rs, ts, zs, es):
        print("es shape:", es.shape)
        print("Rs shape:", Rs.shape)
        print("ts shape:", ts.shape)
        print("zs shape:", zs.shape)
        # Ensure tensors are 4D and concatenate along the batch dimension
        es = es.view(1, -1, 1, 1)
        Rs = Rs.view(1, -1, 1, 1)
        ts = ts.view(1, -1, 1, 1)
        zs = zs.view(1, -1, 1, 1)
        x = torch.cat([Rs, ts, zs, es], dim=1)
        
        # Pass through the initial convolution
        x = self.conv_initial(x)
        
        # Pass through the residual blocks
        x = self.res_blocks(x)
        
        # Pass through the final layers
        x = self.final_layers(x)
        
        return x

'''
The ResBlock3D class represents a 3D residual block. It consists of two 3D convolutional layers (conv1 and conv2) with group normalization (norm1 and norm2) and ReLU activation. The residual connection is implemented using a shortcut connection.
Let's break down the code:

The init method initializes the layers of the residual block.

conv1 and conv2 are 3D convolutional layers with the specified input and output channels, kernel size of 3, and padding of 1.
norm1 and norm2 are group normalization layers with 32 groups and the corresponding number of channels.
If the input and output channels are different, a shortcut connection is created using a 1x1 convolutional layer and group normalization to match the dimensions.


The forward method defines the forward pass of the residual block.

The input x is stored as the residual.
The input is passed through the first convolutional layer (conv1), followed by group normalization (norm1) and ReLU activation.
The output is then passed through the second convolutional layer (conv2) and group normalization (norm2).
If a shortcut connection exists (i.e., input and output channels are different), the residual is passed through the shortcut connection.
The residual is added to the output of the second convolutional layer.
Finally, ReLU activation is applied to the sum.



The ResBlock3D class can be used as a building block in a larger 3D convolutional neural network architecture. It allows for the efficient training of deep networks by enabling the gradients to flow directly through the shortcut connection, mitigating the vanishing gradient problem.
You can create an instance of the ResBlock3D class by specifying the input and output channels:'''
class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(num_groups=32, num_channels=out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = nn.ReLU(inplace=True)(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = nn.ReLU(inplace=True)(out)

        return out
    
    
'''
G3d Class:
- The G3d class represents the 3D convolutional network (G3D) in the diagram.
- It consists of a downsampling path and an upsampling path.

Downsampling Path:
- The downsampling block in the code corresponds to the downsampling path in the diagram.
- It consists of a series of ResBlock3D and 3D average pooling (nn.AvgPool3d) operations.
- The architecture of the downsampling path follows the structure shown in the diagram:
  - ResBlock3D(input_channels, 96) corresponds to the ResBlock3D-96 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-96.
  - ResBlock3D(96, 192) corresponds to the ResBlock3D-192 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-192.
  - ResBlock3D(192, 384) corresponds to the ResBlock3D-384 block.
  - nn.AvgPool3d(kernel_size=2, stride=2) corresponds to the downsampling operation after ResBlock3D-384.
  - ResBlock3D(384, 768) corresponds to the ResBlock3D-768 block.

Upsampling Path:
- The upsampling block in the code corresponds to the upsampling path in the diagram.
- It consists of a series of ResBlock3D and 3D upsampling (nn.Upsample) operations.
- The architecture of the upsampling path follows the structure shown in the diagram:
  - ResBlock3D(768, 384) corresponds to the ResBlock3D-384 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-384.
  - ResBlock3D(384, 192) corresponds to the ResBlock3D-192 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-192.
  - ResBlock3D(192, 96) corresponds to the ResBlock3D-96 block.
  - nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) corresponds to the upsampling operation after ResBlock3D-96.

Final Convolution:
- The final_conv layer in the code corresponds to the GN, ReLU, 3x3x3-Conv3D-96 block in the diagram.
- It takes the output of the upsampling path and applies a 3D convolution with a kernel size of 3 and padding of 1 to produce the final output.

Forward Pass:
- During the forward pass, the input tensor x is passed through the downsampling path, then through the upsampling path, and finally through the final convolution layer.
- The output of the G3d network is a tensor of the same spatial dimensions as the input, but with 96 channels.

In summary, the G3d class in the code aligns well with the 3D convolutional network (G3D) shown in the diagram. The downsampling path, upsampling path, and final convolution layer in the code correspond to the respective blocks in the diagram. The ResBlock3D and pooling/upsampling operations are consistent with the diagram, and the forward pass follows the expected flow of data through the network.
'''


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


class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock2D, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)
        
        identity = self.shortcut(identity)
        
        out += identity
        out = nn.ReLU(inplace=True)(out)
        
        return out
'''
The G2d class consists of the following components:

The input has 96 channels (C96)
The input has a depth dimension of 16 (D16)
The output should have 1536 channels (C1536)

The depth dimension (D16) is present because the input to G2d is a 3D tensor 
(volumetric features) with shape (batch_size, 96, 16, height/4, width/4).
The reshape operation is meant to collapse the depth dimension and increase the number of channels.


The ResBlock2D layers have 512 channels, not 1536 channels as I previously stated. 
The diagram clearly shows 8 ResBlock2D-512 layers before the upsampling blocks that reduce the number of channels.
To summarize, the G2D network takes the orthographically projected 2D feature map from the 3D volumetric features as input.
It first reshapes the number of channels to 512 using a 1x1 convolution layer. 
Then it passes the features through 8 residual blocks (ResBlock2D) that maintain 512 channels. 
This is followed by upsampling blocks that progressively halve the number of channels while doubling the spatial resolution, 
going from 512 to 256 to 128 to 64 channels.
Finally, a 3x3 convolution outputs the synthesized image with 3 color channels.

'''

class G2d(nn.Module):
    def __init__(self, input_channels):
        super(G2d, self).__init__()
        self.reshape = nn.Conv3d(96, 1536, kernel_size=1)  # Reshape C96xD16 → C1536

        self.res_blocks = nn.Sequential(
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
            ResBlock2D(512, 512),
        )

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(512, 256)
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(256, 128)
        )

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(128, 64)
        )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.reshape(x)
        x = self.res_blocks(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.final_conv(x)
        return x


        




'''
In the updated Emtn class, we use two separate networks (head_pose_net and expression_net) to predict the head pose and expression parameters, respectively.

The head_pose_net is a ResNet-18 model pretrained on ImageNet, with the last fully connected layer replaced to output 6 values (3 for rotation and 3 for translation).
The expression_net is another ResNet-18 model with the last fully connected layer adjusted to output the desired dimensions of the expression vector (e.g., 50).

In the forward method, we pass the input x through both networks to obtain the head pose and expression predictions. We then split the head pose output into rotation and translation parameters.
The Emtn module now returns the rotation parameters (Rs, Rd), translation parameters (ts, td), and expression vectors (zs, zd) for both the source and driving images.
Note: Make sure to adjust the dimensions of the rotation, translation, and expression parameters according to your specific requirements and the details provided in the MegaPortraits paper.'''
class Emtn(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_pose_net = resnet18(pretrained=True)
        self.head_pose_net.fc = nn.Linear(512, 6)  # 6 corresponds to rotation and translation parameters
        
        self.expression_net = resnet18(pretrained=False,num_classes=50)  # 50 corresponds to the dimensions of expression vector

    def forward(self, x):
        head_pose = self.head_pose_net(x)
        expression = self.expression_net(x)
        
        # Split head pose into rotation and translation parameters
        rotation = head_pose[:, :3]
        translation = head_pose[:, 3:]
        
        return rotation, translation, expression

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


The volumetric features (vs) obtained from the appearance encoder (Eapp) are warped using the warping field ws2c generated by Ws2c. This warping transforms the volumetric features into the canonical coordinate space, resulting in the canonical volume (vc).
The canonical volume (vc) is then processed by the 3D convolutional network G3d to obtain vc2d.
The vc2d features are warped using the warping field wc2d generated by Wc2d. This warping imposes the driving motion onto the features.
The warped vc2d features are then orthographically projected using average pooling along the depth dimension (denoted as P in the paper). This step reduces the spatial dimensions of the features.
Finally, the projected features (vc2d_projected) are passed through the 2D convolutional network G2d to obtain the final output image (xhat).
'''


class Gbase(nn.Module):
    def __init__(self):
        super(Gbase, self).__init__()
        self.Eapp = Eapp()
        self.Emtn = Emtn()
        self.Ws2c = WarpGenerator() #https://github.com/johndpope/MegaPortrait-hack/issues/5
        self.Wc2d = WarpGenerator()
        self.G3d = G3d(input_channels=96)
        self.G2d = G2d(input_channels=96)

    def forward(self, xs, xd):
        vs, es = self.Eapp(xs)
        Rs, ts, zs = self.Emtn(xs)
        Rd, td, zd = self.Emtn(xd)

        assert vs.shape[1:] == (96, 16, vs.shape[3], vs.shape[4]), f"Expected vs shape (_, 96, 16, H', W'), got {vs.shape}"
        assert es.shape[1] == 512, f"Expected es shape (_, 512), got {es.shape}"
        #assert Rs.shape[1:] == ts.shape[1:] == zs.shape[1:] == (3,), f"Expected Rs, ts, zs shape (_, 3), got {Rs.shape}, {ts.shape}, {zs.shape}"
        
        # Warp volumetric features (vs) using ws2c to obtain canonical volume (vc)
        ws2c = self.Ws2c(Rs, ts, zs, es)
        vc = torch.nn.functional.grid_sample(vs, ws2c)
        assert vc.shape == vs.shape, f"Expected vc shape {vs.shape}, got {vc.shape}"
        
        # Process canonical volume (vc) using G3d to obtain vc2d
        vc2d = self.G3d(vc)
        assert vc2d.shape == vs.shape, f"Expected vc2d shape {vs.shape}, got {vc2d.shape}"
        
        # Warp vc2d using wc2d to impose driving motion
        wc2d = self.Wc2d(Rd, td, zd, es)
        vc2d = torch.nn.functional.grid_sample(vc2d, wc2d)
        assert vc2d.shape == vs.shape, f"Expected vc2d shape {vs.shape}, got {vc2d.shape}"
        
        # Perform orthographic projection (denoted as P in the paper)
        vc2d_projected = torch.mean(vc2d, dim=2)  # Average along the depth dimension
        assert vc2d_projected.shape[1:] == (96, vs.shape[3], vs.shape[4]), f"Expected vc2d_projected shape (_, 96, H/4, W/4), got {vc2d_projected.shape}"
        
        # Pass projected features through G2d to obtain the final output image (xhat)
        xhat = self.G2d(vc2d_projected)
        assert xhat.shape[1:] == (3, xs.shape[2], xs.shape[3]), f"Expected xhat shape (_, 3, H, W), got {xhat.shape}"
        
        return xhat



'''
The high-resolution model (Genh) 
The encoder consists of a series of convolutional layers, residual blocks, and average pooling operations to downsample the input image.
The res_blocks section contains multiple residual blocks operating at the same resolution.
The decoder consists of upsampling operations, residual blocks, and a final convolutional layer to generate the high-resolution output.
'''
class Genh(nn.Module):
    def __init__(self):
        super(Genh, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ResBlock2D(64),
        )
        self.res_blocks = nn.Sequential(
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
            ResBlock2D(64),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock2D(64),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

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


# We load a pre-trained VGG19 network using models.vgg19(pretrained=True) and extract its feature layers using .features. We set the VGG network to evaluation mode and move it to the same device as the input images.
# We define the normalization parameters for the VGG network. The mean and standard deviation values are based on the ImageNet dataset, which was used to train the VGG network. We create tensors for the mean and standard deviation values and move them to the same device as the input images.
# We normalize the input images x and y using the defined mean and standard deviation values. This normalization is necessary to match the expected input format of the VGG network.
# We define the layers of the VGG network to be used for computing the perceptual loss. In this example, we use layers 1, 6, 11, 20, and 29, which correspond to different levels of feature extraction in the VGG network.
# We initialize a variable perceptual_loss to accumulate the perceptual loss values.
# We iterate over the layers of the VGG network using enumerate(vgg). For each layer, we pass the input images x and y through the layer and update their values.
# If the current layer index is in the perceptual_layers list, we compute the perceptual loss for that layer using the L1 loss between the features of x and y. We accumulate the perceptual loss values by adding them to the perceptual_loss variable.
# Finally, we return the computed perceptual loss.

    def perceptual_loss(self, x, y):
        # Load pre-trained VGG network
        vgg = models.vgg19(pretrained=True).features.eval().to(x.device)
        
        # Define VGG normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize input images
        x = (x - mean) / std
        y = (y - mean) / std
        
        # Define perceptual loss layers
        perceptual_layers = [1, 6, 11, 20, 29]
        
        # Initialize perceptual loss
        perceptual_loss = 0.0
        
        # Extract features from VGG network
        for i, layer in enumerate(vgg):
            x = layer(x)
            y = layer(y)
            
            if i in perceptual_layers:
                # Compute perceptual loss for current layer
                perceptual_loss += nn.functional.l1_loss(x, y)
        
        return perceptual_loss

class GHR(nn.Module):
    def __init__(self):
        super(GHR, self).__init__()
        self.Gbase = Gbase()
        self.Genh = Genh()

    def forward(self, xs, xd):
        xhat_base = self.Gbase(xs, xd)
        xhat_hr = self.Genh(xhat_base)
        return xhat_hr



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
The encoder consists of convolutional layers, custom residual blocks, and average pooling operations to downsample the input image.
The SPADE (Spatially-Adaptive Normalization) blocks are applied after the encoder, conditioned on the avatar index.
The decoder consists of custom residual blocks, upsampling operations, and a final convolutional layer to generate the output image.

GDT
'''
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


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.avg_pool(input)
        input = input.view(input.size(0), -1)
        input = self.fc(input)
        return input
    

class Student(nn.Module):
    def __init__(self, num_avatars):
        super(Student, self).__init__()
        self.encoder = nn.Sequential(
            ResNet18(),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 192),
            ResBlock(192, 96),
            ResBlock(96, 48),
            ResBlock(48, 24),
        )
        self.decoder = nn.Sequential(
            SPADEResBlock(24, 48, num_avatars),
            SPADEResBlock(48, 96, num_avatars),
            SPADEResBlock(96, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
            SPADEResBlock(192, 192, num_avatars),
        )
        self.final_layer = nn.Sequential(
            nn.InstanceNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 3, kernel_size=1),
        )

    def forward(self, xd, avatar_index):
        features = self.encoder(xd)
        features = self.decoder(features, avatar_index)
        output = self.final_layer(features)
        return output


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
# Gaze Lossimport torch
# from rt_gene.estimate_gaze_pytorch import GazeEstimator
# import os
# class GazeLoss(nn.Module):
#     def __init__(self, model_path='./models/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'):
#         super(GazeLoss, self).__init__()
#         self.gaze_model = GazeEstimator("cuda:0", [os.path.expanduser(model_path)])
#         # self.gaze_model.eval()
#         self.loss_fn = nn.MSELoss()

#     def forward(self, output_frame, target_frame):
#         output_gaze = self.gaze_model.estimate_gaze_twoeyes(
#             inference_input_left_list=[self.gaze_model.input_from_image(output_frame[0])],
#             inference_input_right_list=[self.gaze_model.input_from_image(output_frame[1])],
#             inference_headpose_list=[[0, 0]]  # Placeholder headpose
#         )

#         target_gaze = self.gaze_model.estimate_gaze_twoeyes(
#             inference_input_left_list=[self.gaze_model.input_from_image(target_frame[0])],
#             inference_input_right_list=[self.gaze_model.input_from_image(target_frame[1])],
#             inference_headpose_list=[[0, 0]]  # Placeholder headpose
#         )

#         loss = self.loss_fn(output_gaze, target_gaze)
#         return loss

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
    
'''
In this updated code:

We define a GazeBlinkLoss module that combines the gaze and blink prediction tasks.
The module consists of a backbone network (VGG-16), a keypoint network, a gaze prediction head, and a blink prediction head.
The backbone network is used to extract features from the left and right eye images separately. The features are then summed to obtain a combined eye representation.
The keypoint network takes the 2D keypoints as input and produces a latent vector of size 64.
The gaze prediction head takes the concatenated eye features and keypoint features as input and predicts the gaze direction.
The blink prediction head takes only the eye features as input and predicts the blink probability.
The gaze loss is computed using both MAE and MSE losses, weighted by w_mae and w_mse, respectively.
The blink loss is computed using binary cross-entropy loss.
The total loss is the sum of the gaze loss and blink loss.

To train this model, you can follow the training procedure you described:

Use Adam optimizer with the specified hyperparameters.
Train for 60 epochs with a batch size of 64.
Use the one-cycle learning rate schedule.
Treat the predictions from RT-GENE and RT-BENE as ground truth.

Note that you'll need to preprocess your data to provide the left eye image, right eye image, 2D keypoints, target gaze, and target blink for each sample during training.
This code provides a starting point for aligning the MediaPipe-based gaze and blink loss with the approach you described. You may need to make further adjustments based on your specific dataset and requirements.

'''
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class GazeBlinkLoss(nn.Module):
    def __init__(self, device, w_mae=15, w_mse=10):
        super(GazeBlinkLoss, self).__init__()
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        self.w_mae = w_mae
        self.w_mse = w_mse
        
        self.backbone = self._create_backbone()
        self.keypoint_net = self._create_keypoint_net()
        self.gaze_head = self._create_gaze_head()
        self.blink_head = self._create_blink_head()
        
    def _create_backbone(self):
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:1])
        return model
    
    def _create_keypoint_net(self):
        return nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    
    def _create_gaze_head(self):
        return nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def _create_blink_head(self):
        return nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, left_eye, right_eye, keypoints, target_gaze, target_blink):
        # Extract eye features using the backbone
        left_features = self.backbone(left_eye)
        right_features = self.backbone(right_eye)
        eye_features = left_features + right_features
        
        # Extract keypoint features
        keypoint_features = self.keypoint_net(keypoints)
        
        # Predict gaze
        gaze_input = torch.cat((eye_features, keypoint_features), dim=1)
        predicted_gaze = self.gaze_head(gaze_input)
        
        # Predict blink
        predicted_blink = self.blink_head(eye_features)
        
        # Compute gaze loss
        gaze_mae_loss = nn.L1Loss()(predicted_gaze, target_gaze)
        gaze_mse_loss = nn.MSELoss()(predicted_gaze, target_gaze)
        gaze_loss = self.w_mae * gaze_mae_loss + self.w_mse * gaze_mse_loss
        
        # Compute blink loss
        blink_loss = nn.BCEWithLogitsLoss()(predicted_blink, target_blink)
        
        # Total loss
        total_loss = gaze_loss + blink_loss
        
        return total_loss, predicted_gaze, predicted_blink
    


# vanilla gazeloss using mediapipe

from typing import Union, List
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MPGazeLoss(object):
    def __init__(self, device):
        self.device = device
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        
    def forward(self, predicted_gaze, target_gaze, face_image):
        # Convert face image from tensor to numpy array
        face_image = face_image.detach().cpu().numpy().transpose(1, 2, 0)
        face_image = (face_image * 255).astype(np.uint8)
        
        # Extract eye landmarks using MediaPipe
        results = self.face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return torch.tensor(0.0).to(self.device)
        
        eye_landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            left_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_LEFT_EYE]
            right_eye_landmarks = [face_landmarks.landmark[idx] for idx in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]
            eye_landmarks.append((left_eye_landmarks, right_eye_landmarks))
        
        # Compute loss for each eye
        loss = 0.0
        for left_eye, right_eye in eye_landmarks:
            # Convert landmarks to pixel coordinates
            h, w = face_image.shape[:2]
            left_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in left_eye]
            right_eye_pixels = [(int(lm.x * w), int(lm.y * h)) for lm in right_eye]
            
            # Create eye mask
            left_mask = torch.zeros((1, h, w)).to(self.device)
            right_mask = torch.zeros((1, h, w)).to(self.device)
            cv2.fillPoly(left_mask[0], [np.array(left_eye_pixels)], 1.0)
            cv2.fillPoly(right_mask[0], [np.array(right_eye_pixels)], 1.0)
            
            # Compute gaze loss for each eye
            left_gaze_loss = F.mse_loss(predicted_gaze * left_mask, target_gaze * left_mask)
            right_gaze_loss = F.mse_loss(predicted_gaze * right_mask, target_gaze * right_mask)
            loss += left_gaze_loss + right_gaze_loss
        
        return loss / len(eye_landmarks)
