
import torch
import torch.nn as nn
from scripts.ecg_dataset_s import ECGDataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class WaveExtractor:
    def __init__(self, weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = UNet()
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def extract_wave(self, image_path):
        dataset = ECGDataset([image_path], transform=self.transform)
        image, _ = dataset[0]  # second item (mask) unused for inference
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            binary_mask = (torch.sigmoid(output) > 0.5).float()

        return binary_mask.cpu().numpy()

    def plot_wave(self, binary_mask, title="Predicted Binary Mask"):
        if binary_mask.ndim == 4:
            mask_2d = binary_mask[0, 0]
        elif binary_mask.ndim == 3:
            mask_2d = binary_mask[0]
        else:
            mask_2d = binary_mask

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_2d, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
