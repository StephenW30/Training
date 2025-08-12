import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class DropBlock2D(nn.Module):
    """
    DropBlock: A regularization method for convolutional networks
    https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self, block_size, keep_prob):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        
    def forward(self, x, training=True):
        if not training or self.keep_prob == 1:
            return x
        
        # Calculate gamma (drop probability)
        n, c, h, w = x.size()
        gamma = ((1. - self.keep_prob) / (self.block_size ** 2)) * \
                ((h * w) / ((h - self.block_size + 1) * (w - self.block_size + 1)))
        
        # Generate mask
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Create valid seed region (avoid dropping at edges to ensure block integrity)
        valid_block_center = torch.zeros_like(x).to(x.device)
        half_block_size = self.block_size // 2
        
        if half_block_size > 0:
            valid_block_center[:, :, half_block_size:h-half_block_size, half_block_size:w-half_block_size] = 1
        else:
            valid_block_center = torch.ones_like(x)
        
        mask = mask * valid_block_center
        
        # Extend mask to block size
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        
        # Invert mask (1 = keep, 0 = drop)
        mask = 1 - mask
        
        # Scale output to maintain the average activation value
        count = torch.sum(mask)
        count = torch.max(count, torch.ones_like(count))
        
        return x * mask * (mask.numel() / count)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Calculate average pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Calculate max pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        concat = torch.cat([avg_pool, max_pool], dim=1)
        # Apply convolution and sigmoid activation
        attention = self.sigmoid(self.conv(concat))
        # Apply attention
        return x * attention


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, block_size=7, keep_prob=0.9, 
                 start_neurons=16, with_attention=False, use_output_activation=True):
        super(UNet, self).__init__()
        self.with_attention = with_attention
        self.use_output_activation = use_output_activation
        
        # Encoder path
        # First layer
        self.enc1_1 = nn.Conv2d(in_channels, start_neurons, 3, padding=1)
        self.drop1_1 = DropBlock2D(block_size, keep_prob)
        self.bn1_1 = nn.BatchNorm2d(start_neurons)
        self.enc1_2 = nn.Conv2d(start_neurons, start_neurons, 3, padding=1)
        self.drop1_2 = DropBlock2D(block_size, keep_prob)
        self.bn1_2 = nn.BatchNorm2d(start_neurons)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second layer
        self.enc2_1 = nn.Conv2d(start_neurons, start_neurons*2, 3, padding=1)
        self.drop2_1 = DropBlock2D(block_size, keep_prob)
        self.bn2_1 = nn.BatchNorm2d(start_neurons*2)
        self.enc2_2 = nn.Conv2d(start_neurons*2, start_neurons*2, 3, padding=1)
        self.drop2_2 = DropBlock2D(block_size, keep_prob)
        self.bn2_2 = nn.BatchNorm2d(start_neurons*2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third layer
        self.enc3_1 = nn.Conv2d(start_neurons*2, start_neurons*4, 3, padding=1)
        self.drop3_1 = DropBlock2D(block_size, keep_prob)
        self.bn3_1 = nn.BatchNorm2d(start_neurons*4)
        self.enc3_2 = nn.Conv2d(start_neurons*4, start_neurons*4, 3, padding=1)
        self.drop3_2 = DropBlock2D(block_size, keep_prob)
        self.bn3_2 = nn.BatchNorm2d(start_neurons*4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck_1 = nn.Conv2d(start_neurons*4, start_neurons*8, 3, padding=1)
        self.drop_bottleneck_1 = DropBlock2D(block_size, keep_prob)
        self.bn_bottleneck_1 = nn.BatchNorm2d(start_neurons*8)
        
        # Spatial attention mechanism (only used when with_attention=True)
        if with_attention:
            self.spatial_attention = SpatialAttention(kernel_size=7)
            
        self.bottleneck_2 = nn.Conv2d(start_neurons*8, start_neurons*8, 3, padding=1)
        self.drop_bottleneck_2 = DropBlock2D(block_size, keep_prob)
        self.bn_bottleneck_2 = nn.BatchNorm2d(start_neurons*8)
        
        # Decoder path
        # Third layer
        self.upconv3 = nn.ConvTranspose2d(start_neurons*8, start_neurons*4, 3, stride=2, padding=1, output_padding=1)
        self.dec3_1 = nn.Conv2d(start_neurons*8, start_neurons*4, 3, padding=1)  # *8 due to concatenation
        self.drop_dec3_1 = DropBlock2D(block_size, keep_prob)
        self.bn_dec3_1 = nn.BatchNorm2d(start_neurons*4)
        self.dec3_2 = nn.Conv2d(start_neurons*4, start_neurons*4, 3, padding=1)
        self.drop_dec3_2 = DropBlock2D(block_size, keep_prob)
        self.bn_dec3_2 = nn.BatchNorm2d(start_neurons*4)
        
        # Second layer
        self.upconv2 = nn.ConvTranspose2d(start_neurons*4, start_neurons*2, 3, stride=2, padding=1, output_padding=1)
        self.dec2_1 = nn.Conv2d(start_neurons*4, start_neurons*2, 3, padding=1)  # *4 due to concatenation
        self.drop_dec2_1 = DropBlock2D(block_size, keep_prob)
        self.bn_dec2_1 = nn.BatchNorm2d(start_neurons*2)
        self.dec2_2 = nn.Conv2d(start_neurons*2, start_neurons*2, 3, padding=1)
        self.drop_dec2_2 = DropBlock2D(block_size, keep_prob)
        self.bn_dec2_2 = nn.BatchNorm2d(start_neurons*2)
        
        # First layer
        self.upconv1 = nn.ConvTranspose2d(start_neurons*2, start_neurons, 3, stride=2, padding=1, output_padding=1)
        self.dec1_1 = nn.Conv2d(start_neurons*2, start_neurons, 3, padding=1)  # *2 due to concatenation
        self.drop_dec1_1 = DropBlock2D(block_size, keep_prob)
        self.bn_dec1_1 = nn.BatchNorm2d(start_neurons)
        self.dec1_2 = nn.Conv2d(start_neurons, start_neurons, 3, padding=1)
        self.drop_dec1_2 = DropBlock2D(block_size, keep_prob)
        self.bn_dec1_2 = nn.BatchNorm2d(start_neurons)
        
        # Output layer
        self.output = nn.Conv2d(start_neurons, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def check_size(self, x, target_size):
        """Check and adjust feature map dimensions to match target size"""
        if x.size()[2:] != target_size:
            # Calculate required padding
            diff_h = target_size[0] - x.size(2)
            diff_w = target_size[1] - x.size(3)
            
            # Apply padding or cropping
            if diff_h > 0 or diff_w > 0:
                # Need padding
                pad_h1 = diff_h // 2
                pad_h2 = diff_h - pad_h1
                pad_w1 = diff_w // 2
                pad_w2 = diff_w - pad_w1
                x = F.pad(x, [pad_w1, pad_w2, pad_h1, pad_h2])
            elif diff_h < 0 or diff_w < 0:
                # Need cropping
                crop_h1 = -diff_h // 2
                crop_h2 = x.size(2) - (-diff_h - crop_h1)
                crop_w1 = -diff_w // 2
                crop_w2 = x.size(3) - (-diff_w - crop_w1)
                x = x[:, :, crop_h1:crop_h2, crop_w1:crop_w2]
        return x
        
    def forward(self, x, training=True):
        # Store output dimensions of each layer for later use
        sizes = []
        
        # Encoder
        # First layer
        enc1 = self.enc1_1(x)
        enc1 = self.drop1_1(enc1, training)
        enc1 = F.relu(self.bn1_1(enc1))
        enc1 = self.enc1_2(enc1)
        enc1 = self.drop1_2(enc1, training)
        enc1 = F.relu(self.bn1_2(enc1))
        sizes.append(enc1.size()[2:])  # Record dimensions
        pool1 = self.pool1(enc1)
        
        # Second layer
        enc2 = self.enc2_1(pool1)
        enc2 = self.drop2_1(enc2, training)
        enc2 = F.relu(self.bn2_1(enc2))
        enc2 = self.enc2_2(enc2)
        enc2 = self.drop2_2(enc2, training)
        enc2 = F.relu(self.bn2_2(enc2))
        sizes.append(enc2.size()[2:])  # Record dimensions
        pool2 = self.pool2(enc2)
        
        # Third layer
        enc3 = self.enc3_1(pool2)
        enc3 = self.drop3_1(enc3, training)
        enc3 = F.relu(self.bn3_1(enc3))
        enc3 = self.enc3_2(enc3)
        enc3 = self.drop3_2(enc3, training)
        enc3 = F.relu(self.bn3_2(enc3))
        sizes.append(enc3.size()[2:])  # Record dimensions
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck_1(pool3)
        bottleneck = self.drop_bottleneck_1(bottleneck, training)
        bottleneck = F.relu(self.bn_bottleneck_1(bottleneck))
        
        # Apply spatial attention mechanism if specified
        if self.with_attention:
            bottleneck = self.spatial_attention(bottleneck)
            
        bottleneck = self.bottleneck_2(bottleneck)
        bottleneck = self.drop_bottleneck_2(bottleneck, training)
        bottleneck = F.relu(self.bn_bottleneck_2(bottleneck))
        
        # Decoder
        # Third layer
        up3 = self.upconv3(bottleneck)
        # Adjust up3 size to match enc3
        up3 = self.check_size(up3, sizes[2])
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3_1(merge3)
        dec3 = self.drop_dec3_1(dec3, training)
        dec3 = F.relu(self.bn_dec3_1(dec3))
        dec3 = self.dec3_2(dec3)
        dec3 = self.drop_dec3_2(dec3, training)
        dec3 = F.relu(self.bn_dec3_2(dec3))
        
        # Second layer
        up2 = self.upconv2(dec3)
        # Adjust up2 size to match enc2
        up2 = self.check_size(up2, sizes[1])
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2_1(merge2)
        dec2 = self.drop_dec2_1(dec2, training)
        dec2 = F.relu(self.bn_dec2_1(dec2))
        dec2 = self.dec2_2(dec2)
        dec2 = self.drop_dec2_2(dec2, training)
        dec2 = F.relu(self.bn_dec2_2(dec2))
        
        # First layer
        up1 = self.upconv1(dec2)
        # Adjust up1 size to match enc1
        up1 = self.check_size(up1, sizes[0])
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1_1(merge1)
        dec1 = self.drop_dec1_1(dec1, training)
        dec1 = F.relu(self.bn_dec1_1(dec1))
        dec1 = self.dec1_2(dec1)
        dec1 = self.drop_dec1_2(dec1, training)
        dec1 = F.relu(self.bn_dec1_2(dec1))
        
        # Output
        out = self.output(dec1)
        
        # Ensure output matches input dimensions
        out = self.check_size(out, x.size()[2:])
        
        # Apply activation function only if specified
        if self.use_output_activation:
            out = self.sigmoid(out)
        
        return out


def create_backbone_model(input_size=(1500, 1500, 3), block_size=7, keep_prob=0.9, 
                      start_neurons=16, use_output_activation=True):
    """Create base U-Net model (without spatial attention)"""
    return UNet(
        in_channels=input_size[2], 
        out_channels=1,
        block_size=block_size,
        keep_prob=keep_prob,
        start_neurons=start_neurons,
        with_attention=False,
        use_output_activation=use_output_activation
    )


def create_sa_unet_model(input_size=(1500, 1500, 3), block_size=7, keep_prob=0.9, 
                        start_neurons=16, use_output_activation=True):
    """Create U-Net model with spatial attention"""
    return UNet(
        in_channels=input_size[2], 
        out_channels=1,
        block_size=block_size,
        keep_prob=keep_prob,
        start_neurons=start_neurons,
        with_attention=True,
        use_output_activation=use_output_activation
    )


# Adjust model to support 1x1x1500x1500 input
def create_sa_unet_model_for_single_channel(block_size=7, keep_prob=0.9, 
                                           start_neurons=16, use_output_activation=True):
    """Create U-Net model with spatial attention for single-channel input"""
    return UNet(
        in_channels=1,  # Modified for 1-channel input
        out_channels=1,
        block_size=block_size,
        keep_prob=keep_prob,
        start_neurons=start_neurons,
        with_attention=True,
        use_output_activation=use_output_activation
    )
