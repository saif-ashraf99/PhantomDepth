import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from depth_midas import estimate_depth_midas

def train_conditional_gan(
    generator, 
    discriminator, 
    dataloader,
    midas,
    midas_transform,
    device='cpu',
    lr=1e-4,
    num_epochs=5,
    lambda_gan=1.0,
    lambda_l1=10.0
):
    """
    Trains the conditional GAN on a dataset of single images.
    The idea is:
      1) For each image, run MiDaS to get an initial depth estimate.
      2) Generator refines the depth.
      3) Discriminator classifies real/fake (if ground truth depth is available).
    """
    g_opt = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    adversarial_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            img = batch_data['image'].to(device)       # (N,3,H,W)
            depth_gt = batch_data['depth_gt']          # (N,1,H,W) or None

            # 1) Estimate depth with MiDaS (per image)
            init_depth_list = []
            for b in range(img.size(0)):
                # Convert each image back to numpy HWC for MiDaS
                img_np = (img[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                depth_pred = estimate_depth_midas(midas, midas_transform, img_np, device=device)
                init_depth_list.append(torch.from_numpy(depth_pred).unsqueeze(0))  # (1,H,W)

            init_depth = torch.stack(init_depth_list, dim=0).to(device)  # (N,1,H,W), [0,1]

            # 2) Generator forward pass
            gen_input = torch.cat([img, init_depth], dim=1)  # (N,4,H,W)
            refined_depth = generator(gen_input)

            # 3) Discriminator updates
            d_opt.zero_grad()

            # Real pass (only if we have ground-truth depth)
            if depth_gt is not None:
                real_input = torch.cat([img, init_depth, depth_gt.to(device)], dim=1)  # (N,5,H,W)
                real_pred = discriminator(real_input)
                real_loss = adversarial_criterion(real_pred, torch.ones_like(real_pred))
            else:
                real_loss = 0.0

            # Fake pass
            fake_input = torch.cat([img, init_depth, refined_depth.detach()], dim=1)   # (N,5,H,W)
            fake_pred = discriminator(fake_input)
            fake_loss = adversarial_criterion(fake_pred, torch.zeros_like(fake_pred))

            if depth_gt is not None:
                d_loss = (real_loss + fake_loss) * 0.5
            else:
                d_loss = fake_loss  # If no GT, we have no "real" examples

            d_loss.backward()
            d_opt.step()

            # 4) Generator updates
            g_opt.zero_grad()
            fake_input_g = torch.cat([img, init_depth, refined_depth], dim=1)
            fake_pred_g = discriminator(fake_input_g)
            g_adv_loss = adversarial_criterion(fake_pred_g, torch.ones_like(fake_pred_g))

            if depth_gt is not None:
                l1_loss_val = l1_criterion(refined_depth, depth_gt.to(device))
            else:
                # Without GT, you might consider other constraints (smoothness, etc.)
                l1_loss_val = 0.0

            g_loss = lambda_gan * g_adv_loss + lambda_l1 * l1_loss_val
            g_loss.backward()
            g_opt.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] "
                      f"D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f} "
                      f"(Adv: {g_adv_loss:.4f}, L1: {l1_loss_val:.4f})")
