import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SingleImageDataset(Dataset):
    """
    Example dataset that:
      - Loads an RGB image
      - Optionally loads a ground-truth depth map if available
    You can adapt this to your own folder structure or COCO-like format.
    """
    def __init__(self, image_folder, depth_folder=None, transform=None):
        super().__init__()
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
        
        self.depth_paths = None
        if depth_folder is not None and os.path.exists(depth_folder):
            self.depth_paths = sorted(glob.glob(os.path.join(depth_folder, '*.png')))
            # Ensure matching image-depth pairs if you have a 1-to-1 correspondence

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        if self.transform:
            img = self.transform(img)  # e.g., transforms.ToTensor()

        depth_gt = None
        if self.depth_paths is not None:
            depth_path = self.depth_paths[idx]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = depth_img.astype(np.float32) / 255.0  # Scale to [0,1]
            depth_gt = torch.from_numpy(depth_img).unsqueeze(0)  # (1,H,W)

        return {
            'image': img,         # (3,H,W) float
            'depth_gt': depth_gt  # (1,H,W) float or None
        }
