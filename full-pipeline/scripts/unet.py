import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1     = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.bn1       = nn.BatchNorm2d(out_channels)
        self.relu      = nn.ReLU(inplace=True)
        self.conv2     = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2       = nn.BatchNorm2d(out_channels)
        self.residual  = nn.Conv2d(in_channels,  out_channels, 1)
    def forward(self, x):
        identity = self.residual(x)
        out      = self.relu(self.bn1(self.conv1(x)))
        out      = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1      = ResidualBlock(1,   64);    self.pool1 = nn.MaxPool2d(2)
        self.down2      = ResidualBlock(64,  128);   self.pool2 = nn.MaxPool2d(2)
        self.down3      = ResidualBlock(128, 256);   self.pool3 = nn.MaxPool2d(2)
        self.down4      = ResidualBlock(256, 512);   self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(512, 1024)
        self.up4        = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4       = ResidualBlock(512+512, 512)
        self.up3        = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3       = ResidualBlock(256+256, 256)
        self.up2        = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2       = ResidualBlock(128+128, 128)
        self.up1        = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec1       = ResidualBlock(64+64,   64)
        self.final      = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))
        u4 = self.up4(bn);  u4 = self.dec4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4);  u3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3);  u2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2);  u1 = self.dec1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.final(u1))
