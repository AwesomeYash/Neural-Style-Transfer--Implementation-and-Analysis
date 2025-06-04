"""
Neural Style Transfer using VGG19

Name: Priyanshu Ranka (NUID: 002035396)
Professor: Prof. Bruce Maxwell
Course: CS 5330 - Pattern Recognition and Computer Vision
Semester: Spring 2025
Description: This script contains utility functions for image processing, loss calculation, and visualization.
"""
# Import necessary libraries
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import time
# Import necessary libraries
import torch
from torch.nn.functional import mse_loss


# Define constants for normalization
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

normalize = T.Normalize(mean=MEAN, std=STD)
denormalize = T.Normalize(mean=[-m/s for m, s in zip(MEAN, STD)],
                          std=[1/std for std in STD])

# Define a function to get the tensor transformer
def get_transformer(imsize=None, cropsize=None):
    """Get a tensor transformer."""
    transformer = []
    if imsize:
        transformer.append(T.Resize(imsize))
    if cropsize:
        transformer.append(T.CenterCrop(cropsize))
    transformer.append(T.ToTensor())
    transformer.append(normalize)
    return T.Compose(transformer)

# Define functions to load images
def imload(path, imsize=None, cropsize=None):
    """Load a image."""
    transformer = get_transformer(imsize=imsize, cropsize=cropsize)
    image = Image.open(path).convert("RGB")
    return transformer(image).unsqueeze(0)

# Define a function to save images
def imsave(image, save_path):
    """Save a image."""
    image = denormalize(torchvision.utils.make_grid(image)).clamp_(0.0, 1.0)
    torchvision.utils.save_image(image, save_path)
    return None

# Define a function to calculate content loss
def contentLoss(features, targets, nodes):
    """Calculate Content Loss."""
    content_loss = 0
    for node in nodes:
        content_loss += mse_loss(features[node], targets[node])
    return content_loss

# Define a function to calculate gram matrix
def gram(x):
    """Transfer a feature to gram matrix."""
    b, c, h, w = x.size()
    f = x.flatten(2)
    g = torch.bmm(f, f.transpose(1, 2))
    return g.div(h*w)

# Define a function to calculate style loss
def gramLoss(features, targets, nodes):
    """Calcuate Gram Loss."""
    gram_loss = 0
    for node in nodes:
        gram_loss += mse_loss(gram(features[node]), gram(targets[node]))
    return gram_loss

# Define a function to calculate total variation loss
def tvLoss(x):
    """Calc Total Variation Loss."""
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss


# Plot loss curves from metrics file
def plot_losses(metrics_file, save_path=None, show=False):
    """Plot loss curves from metrics file."""
    if isinstance(metrics_file, str) or isinstance(metrics_file, Path):
        if str(metrics_file).endswith('.json'):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        elif str(metrics_file).endswith('.npy'):
            metrics = np.load(metrics_file, allow_pickle=True).item()
        else:
            raise ValueError("Metrics file must be .json or .npy")
    else:
        # Assume metrics is already a dictionary
        metrics = metrics_file
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Plot each loss
    iterations = range(1, len(metrics['total_loss']) + 1)
    
    loss_types = ['total_loss', 'content_loss', 'style_loss', 'tv_loss']
    titles = ['Total Loss', 'Content Loss', 'Style Loss', 'TV Loss']
    
    for ax, loss_type, title in zip(axes, loss_types, titles):
        ax.plot(iterations, metrics[loss_type])
        ax.set_title(title)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()
