import math
import torch
from torch import nn
import torch.nn.functional as F


class SRGAN_G(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(SRGAN_G, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        self.resblocks = []  
        for _ in range(16):  
            self.resblocks.append(ResidualBlock(64))  
        self.resblocks = nn.Sequential(*self.resblocks) 
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        
        self.block3 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.block3.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block3 = nn.Sequential(*self.block3)
        

    def forward(self, x):
        h1 = self.block1(x)
        temp = h1
        h2 = self.resblocks(h1)
        h3 = self.block2(h2)
        h3 = h3 + temp
        h4 = self.block3(h3)

        return (torch.tanh(h4) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.lrelu(residual)

        return x + residual

class ResidualDenseBlock(nn.Module):
    def __init__(self,in_channel = 64, inc_channel = 32, beta = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel + 2*inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel + 3*inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channel + 4*inc_channel, in_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b = beta
    
    def forward(self, x):
        h1 = self.lrelu(self.conv1(x))
        h2 = self.lrelu(self.conv2(torch.cat((x, h1), dim = 1)))
        h3 = self.lrelu(self.conv3(torch.cat((x, h1, h2), dim = 1)))
        h4 = self.lrelu(self.conv4(torch.cat((x, h1, h2, h3), dim = 1)))
        out = self.conv5(torch.cat((x, h1, h2, h3, h4), dim = 1))

        return x + self.b * out

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self,in_channel = 64, out_channel = 32, beta = 0.2):
        super().__init__()
        self.rdb = ResidualDenseBlock(in_channel, out_channel)
        self.b = beta
    
    def forward(self, x):
        out = self.rdb(x)
        out = self.rdb(out)
        out = self.rdb(out)

        return x + self.b * out
    
class Generator(nn.Module):
    def __init__(self, scale_factor):

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        )
        
        self.rrdb = []
        for _ in range(18):  
            self.rrdb.append(ResidualInResidualDenseBlock(64, 32, 0.2))  
        self.rrdb = nn.Sequential(*self.rrdb)
        
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.trunk_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

        self.resblocks1 = []
        for _ in range(3):  
            self.resblocks1.append(ResidualBlock2(64))  
        self.resblocks1 = nn.Sequential(*self.resblocks1) 

        self.resblocks2 = []
        for _ in range(3):  
            self.resblocks2.append(ResidualBlock2(64))  
        self.resblocks2 = nn.Sequential(*self.resblocks2) 
        
        self.block4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        h1 = self.block1(x)
        temp = h1
        h2 = self.trunk_conv(self.rrdb(h1))
        h2 = h2 + temp
        h3 = F.interpolate(h2, scale_factor=2, mode='nearest')
        h4 = self.resblocks1(h3)
        h5 = F.interpolate(h4, scale_factor=2, mode='nearest')
        h6 = self.resblocks2(h5)
       
        h7 = self.block4(h6)

        return (torch.tanh(h7) + 1) / 2

class ESRGAN_G(nn.Module):
    def __init__(self, scale_factor):

        super(ESRGAN_G, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        )
        
        self.rrdb = []
        for _ in range(23):  
            self.rrdb.append(ResidualInResidualDenseBlock(64, 32, 0.2))  
        self.rrdb = nn.Sequential(*self.rrdb)
        
        self.trunk_conv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        

        self.block4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        

    def forward(self, x):
        h1 = self.block1(x)
        temp = h1
        h2 = self.trunk_conv2(self.trunk_conv(self.rrdb(h1)))
        h2 = h2 + temp
        h3 = self.lrelu(self.upconv1(F.interpolate(h2, scale_factor=2, mode='nearest')))
        h4 = self.lrelu(self.upconv2(F.interpolate(h3, scale_factor=2, mode='nearest')))
        h5 = self.block4(self.lrelu(self.HRconv(h4)))

        return (torch.tanh(h5) + 1) / 2


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
