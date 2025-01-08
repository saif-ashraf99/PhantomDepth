import torch.nn as nn

class Generator(nn.Module):
    """
    A simple U-Net-like generator that takes:
     - RGB image (3-channel)
     - initial depth estimate (1-channel)
    and outputs a refined depth map (1-channel).
    """
    def __init__(self, input_channels=4, output_channels=1):
        super().__init__()

        # Down-sampling
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Up-sampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)  
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        u1 = self.up1(d3)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        return u3


class Discriminator(nn.Module):
    """
    A simple PatchGAN-like discriminator that sees:
     - input RGB + initial depth (4 channels)
     - predicted/refined depth (1 channel)
    => total 5 input channels
    and outputs a real/fake score map.
    """
    def __init__(self, input_channels=5):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 1)  # PatchGAN output
        )

    def forward(self, x):
        return self.main(x)
