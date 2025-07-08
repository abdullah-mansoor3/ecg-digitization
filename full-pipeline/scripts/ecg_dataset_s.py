
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class ECGDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        # Dummy mask to keep output format same as training dataset
        mask = image.clone()
        return image, mask
