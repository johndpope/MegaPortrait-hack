import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class RCCA(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.num_heads = num_heads
        self.out_channels = out_channels
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape to (batch_size, num_heads, height, width, -1)
        query = query.view(batch_size, self.num_heads, -1, height, width).permute(0, 1, 3, 4, 2)
        key = key.view(batch_size, self.num_heads, -1, height, width).permute(0, 1, 3, 4, 2)
        value = value.view(batch_size, self.num_heads, -1, height, width).permute(0, 1, 3, 4, 2)
        
        # Compute attention weights
        energy = torch.matmul(query, key.transpose(-2, -1))
        attention = torch.softmax(energy, dim=-1)
        
        # Attend to values
        out = torch.matmul(attention, value).permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(batch_size, self.out_channels, height, width)
        
        return out + x

class CCNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, num_recurrence=2):
        super().__init__()
        self.rcca = RCCA(in_channels, out_channels, num_heads)
        self.num_recurrence = num_recurrence
        
    def forward(self, x):
        for _ in range(self.num_recurrence):
            x = self.rcca(x)
        return x

class CCNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, num_recurrence=2):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=None,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        self.ccnet = CCNet(512, 512, num_heads, num_recurrence)
        
    def forward(self, x):
        x = self.unet.conv_in(x)
        x = self.unet.time_embed(self.unet.timestep_embedding(torch.zeros((x.shape[0],)).to(x.dtype).to(x.device)))
        down_block_res_samples = self.unet.down_blocks[0](x, x)
        for down_block in self.unet.down_blocks[1:]:
            down_block_res_samples = down_block(down_block_res_samples, x)
        
        down_block_res_samples = self.ccnet(down_block_res_samples)
        
        for up_block in self.unet.up_blocks:
            down_block_res_samples = up_block(down_block_res_samples, x)
            
        return self.unet.conv_out(down_block_res_samples)
    

class CCNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1, num_recurrence=2):
        super(CCNet3D, self).__init__()
        self.ccnet = RCCA3D(in_channels, out_channels, num_heads)
        self.num_recurrence = num_recurrence

    def forward(self, x):
        for _ in range(self.num_recurrence):
            x = self.ccnet(x)
        return x

class RCCA3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=1):
        super(RCCA3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.num_heads = num_heads
        self.out_channels = out_channels

    def forward(self, x):
        batch_size, _, depth, height, width = x.size()

        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Reshape to (batch_size, num_heads, depth, height, width, -1)
        query = query.view(batch_size, self.num_heads, -1, depth, height, width).permute(0, 1, 3, 4, 5, 2)
        key = key.view(batch_size, self.num_heads, -1, depth, height, width).permute(0, 1, 3, 4, 5, 2)
        value = value.view(batch_size, self.num_heads, -1, depth, height, width).permute(0, 1, 3, 4, 5, 2)

        # Compute attention weights
        energy = torch.matmul(query, key.transpose(-2, -1))
        attention = torch.softmax(energy, dim=-1)

        # Attend to values
        out = torch.matmul(attention, value).permute(0, 1, 5, 2, 3, 4).contiguous()
        out = out.view(batch_size, self.out_channels, depth, height, width)

        return out + x