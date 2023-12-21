'''Обучение модели
'''

import torch
import pyredner

from create_model import create_model

# Set requires_grad=True since we want to optimize them later
cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277], requires_grad=True)
cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918], requires_grad=True)
shape_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
color_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
ambient_color = torch.ones(3, device=pyredner.get_device(), requires_grad=True)
dir_light_intensity = torch.zeros(3, device=pyredner.get_device(), requires_grad=True)

# Use two different optimizers for different learning rates
optimizer = torch.optim.Adam([
    shape_coeffs, 
    color_coeffs, 
    ambient_color, 
    dir_light_intensity
], lr=0.1)

cam_optimizer = torch.optim.Adam([cam_pos, cam_look_at], lr=0.5)


def train_model(target,
                shape_mean, 
                shape_basis, 
                triangle_list, 
                color_mean, 
                color_basis,
                num_iters=500):
    
    imgs, losses = [], []

    for t in range(num_iters):
        optimizer.zero_grad()
        cam_optimizer.zero_grad()

        img = create_model(
            cam_pos, 
            cam_look_at, 
            shape_coeffs, 
            color_coeffs, 
            ambient_color, 
            dir_light_intensity,
            shape_mean, 
            shape_basis, 
            triangle_list, 
            color_mean, 
            color_basis
        )

        # Compute the loss function. Here it is L2 plus a regularization 
        # term to avoid coefficients to be too far from zero.
        # Both img and target are in linear color space, 
        # so no gamma correction is needed.

        loss = (img - target).pow(2).mean()
        loss = loss + 0.0001 * shape_coeffs.pow(2).mean() + 0.001 * color_coeffs.pow(2).mean()
        loss.backward()

        optimizer.step()
        cam_optimizer.step()

        ambient_color.data.clamp_(0.0)
        dir_light_intensity.data.clamp_(0.0)

        losses.append(loss.data.item())

        # Only store images every 10th iterations
        if t % 10 == 0:
            # Record the Gamma corrected image
            imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu()) 
    
    return imgs, losses