import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

from datasets import SingleImageDataset
from depth_midas import load_midas_model, estimate_depth_midas
from models import Generator, Discriminator
from trainer import train_conditional_gan
from pointcloud import depth_to_pointcloud
import open3d as o3d

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # -------------------------
    # 1) Prepare dataset
    # -------------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = SingleImageDataset(
        image_folder='data/images',     # Update with your images folder
        depth_folder='data/depth',      # (Optional) If you have GT depth
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # -------------------------
    # 2) Load MiDaS for initial depth
    # -------------------------
    midas, midas_transform = load_midas_model(device)

    # -------------------------
    # 3) Create our Generator & Discriminator
    # -------------------------
    generator = Generator(input_channels=4, output_channels=1).to(device)
    discriminator = Discriminator(input_channels=5).to(device)

    # -------------------------
    # 4) Train (optional, you can skip if you only want inference)
    # -------------------------
    train_conditional_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        midas=midas,
        midas_transform=midas_transform,
        device=device,
        lr=2e-4,
        num_epochs=5
    )

    # -------------------------
    # 5) Inference on a single test image -> 3D visualization
    # -------------------------
    test_img_path = 'data/test_img.jpg'
    img_cv = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 5a) Estimate depth using MiDaS
    init_depth_np = estimate_depth_midas(midas, midas_transform, img_rgb, device=device)
    init_depth_tensor = torch.from_numpy(init_depth_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    # 5b) Run generator
    img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)  # (1,3,H,W)
    gen_input = torch.cat([img_tensor, init_depth_tensor], dim=1)  # (1,4,H,W)
    generator.eval()
    with torch.no_grad():
        refined_depth_tensor = generator(gen_input)

    refined_depth_np = refined_depth_tensor.squeeze().cpu().numpy()  # (H,W), in [0,1]

    # 5c) Convert refined depth to point cloud
    pcd = depth_to_pointcloud(img_rgb, refined_depth_np, fx=500, fy=500)

    # 5d) Visualize with Open3D
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
