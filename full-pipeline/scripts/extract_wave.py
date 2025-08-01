import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from scripts.unet import UNet

class WaveExtractor:
    def __init__(self, weights_path, device='cpu'):
        """
        Initialize the UNet model and set device.
        """
        print('initializing wave extractor...')
        self.device = torch.device(device)
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        print('Wave extractor initialized.')

        # same IMG_SIZE as in training
        self.input_size = (512, 1024)   # (height, width)
        # build the torchvision transform
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),          # output in [0,1], shape [1,H,W]
        ])

    @staticmethod
    def center_crop(tensor, target_h):
        # tensor shape: [1, H, W]
        _, h, w = tensor.shape
        top = (h - target_h) // 2
        return tensor[:, top:top+target_h, :]

    def extract_wave(self, cropped_lead_img_path):
        """
        Takes the path to a cropped lead image, returns the extracted wave mask as a numpy array.
        The output is center-cropped (in case of padding) and resized back to original.
        """
        # load grayscale
        img = Image.open(cropped_lead_img_path).convert("L")
        orig_w, orig_h = img.size

        # preprocess
        inp = self.transform(img).unsqueeze(0).to(self.device)  # [1,1,512,1024]

        with torch.no_grad():
            out = self.model(inp)                          # [1,1,512,1024], sigmoid already applied
            mask = (out > 0.5).float().cpu()[0,0]          # [512,1024]

        # if original image had padding in height, center-crop back
        # if mask.shape[0] > orig_h:
        #     mask = self.center_crop(mask.unsqueeze(0), orig_h)[0]

        # now resize mask back to original size
        mask_np = mask.numpy().astype(np.uint8)
        mask_pil = Image.fromarray(mask_np * 255)           # back to 0–255
        # mask_resized = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)

        return binary_mask

    def plot_wave(self, binary_mask, title="Extracted Waveform Binary Mask"):
        """
        Plots the binary waveform mask.
        Args:
            binary_mask (np.ndarray): 2D binary mask (output of extract_wave)
            title (str): Plot title
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(binary_mask, cmap='gray', aspect='auto')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
