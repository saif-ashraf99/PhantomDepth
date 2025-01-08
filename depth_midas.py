import torch
import numpy as np

def load_midas_model(device='cpu'):
    """
    Downloads/loads a MiDaS model from Torch Hub if not already present,
    sets it to evaluation mode.
    """
    model_type = "DPT_Large"  # or "DPT_Hybrid", "MiDaS_small", etc.
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform


def estimate_depth_midas(midas, transform, image, device='cpu'):
    """
    Given a midas model and transform, estimate depth for a single image (PIL or np.array).
    image is expected to be a numpy array in RGB or a PIL.Image in RGB.
    Returns a numpy array with depth estimates normalized to [0,1].
    """
    import PIL.Image as Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # convert np.array -> PIL if needed

    input_batch = transform(image).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # (width, height) -> reversed
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth = prediction.cpu().numpy()
    # Normalize to [0,1]
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    return depth_norm
