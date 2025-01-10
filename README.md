# PhantomDepth — Generative Single-View 3D Reconstruction

**PhantomDepth** is a project that demonstrates how to perform single-view 3D reconstruction by combining:

- **Pretrained depth estimation** (MiDaS)  
- **Generative Adversarial Networks (GANs)** for refining depth  
- **3D point cloud generation** for visualization  

The project is structured into multiple files (for datasets, models, trainer, etc.) to keep the code organized and modular.

---

## Project Structure

```
single_view_3d/
├── data/
│   ├── images/
│   ├── depth/
│   └── test_img.jpg
├── datasets.py
├── depth_midas.py
├── models.py
├── pointcloud.py
├── trainer.py
├── main.py
├── requirements.txt
└── README.md
```

### Directory Contents

1. **`data/`**  
   - `images/`: Holds your RGB input images.  
   - `depth/`: (Optional) Holds ground-truth depth maps, if available.  
   - `test_img.jpg`: An example image you can use to test the pipeline.

2. **`datasets.py`**  
   - Contains the `SingleImageDataset` class for loading images (and optional depth maps).

3. **`depth_midas.py`**  
   - Loads a pretrained MiDaS model from Torch Hub (e.g., DPT-Large).  
   - Provides a function to estimate depth for a single image.

4. **`models.py`**  
   - Defines the **Generator** (a simple U-Net-like network) and **Discriminator** (PatchGAN-style) used for conditional GAN training.

5. **`pointcloud.py`**  
   - Contains a helper function to convert an RGB image and its corresponding depth map into a 3D point cloud (using Open3D).

6. **`trainer.py`**  
   - Provides the main training loop for the conditional GAN.  
   - Uses MiDaS to get initial depth estimates, then trains the Generator to refine them, with the Discriminator enforcing adversarial feedback.

7. **`main.py`**  
   - The primary entry point. It loads the dataset, the MiDaS model, creates generator/discriminator, trains them (if ground-truth depth is present), and finally runs inference on a test image to visualize the 3D reconstruction.

8. **`requirements.txt`**  
   - Lists recommended Python packages (torch, torchvision, timm, open3d, etc.) with example version pins.

9. **`README.md`**  
   - The file you are reading right now, with instructions and an overview.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/saif-ashraf99/PhantomDepth
   cd PhantomDepth
   ```

2. **Create a Python environment** (optional but recommended):

   ```bash
   conda create -n phantomdepth_env python=3.9
   conda activate phantomdepth_env
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have a compatible GPU and CUDA setup if you plan to use CUDA.

---

## Usage

1. **Prepare Your Data**  
   - Place all your RGB images in `data/images/`.  
   - If you have **ground-truth depth maps**, place them in `data/depth/` with matching filenames (e.g., `image_001.jpg` and `image_001.png`).  
     - If no ground-truth depth is available, you can still run the code, but the discriminator will only see “fake” samples (you may need alternative/self-supervised losses).

2. **Run the Pipeline**  
   From the `PhantomDepth` folder, execute:
   ```bash
   python main.py
   ```

   **What happens?**  
   - **Training**: If you have GT depth maps, the script will train a conditional GAN for a few epochs (default is 5).  
   - **Inference**: After training, `main.py` loads a test image (`data/test_img.jpg`), uses MiDaS to get initial depth, refines it with the trained generator, then converts it to a 3D point cloud and displays it in an Open3D viewer.

3. **View Results**  
   - The console will show training logs, including **Generator** and **Discriminator** loss values.  
   - After training, an Open3D window should pop up with the reconstructed point cloud for `test_img.jpg`.  
     - Use your mouse/trackpad to rotate and inspect the 3D reconstruction.

---

## Customization & Next Steps

- **Camera Intrinsics**: The default `fx=500, fy=500` in `depth_to_pointcloud()` (in `pointcloud.py`) is purely illustrative. If you have real camera intrinsics, replace them to get more accurate 3D point geometry.
- **Longer Training**: Increase `num_epochs` in `main.py` if you have a larger dataset. You may also want to lower the batch size if you run out of GPU memory.
- **Enhanced Architectures**: 
  - Try a more sophisticated U-Net architecture for the **Generator** or experiment with attention mechanisms.  
  - Use a different patch size or multi-scale approach for the **Discriminator**.
- **Loss Functions**:  
  - Add **perceptual loss**, **smoothness constraints**, or **SSIM** for more realistic depth maps.  
  - If no GT depth is available, look into **self-supervised** or **monocular depth** consistency losses.
- **Explore NeRF**: For more advanced volumetric 3D reconstruction, check out [instant-ngp](https://github.com/NVlabs/instant-ngp) or other NeRF frameworks. This project is a simpler depth-map-based approach.

---