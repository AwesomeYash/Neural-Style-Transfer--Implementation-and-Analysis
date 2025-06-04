"""
Neural Style Transfer using VGG19

Name: Priyanshu Ranka (NUID: 002035396)
Professor: Prof. Bruce Maxwell
Course: CS 5330 - Pattern Recognition and Computer Vision
Semester: Spring 2025
Descriiption: The Main script for running the neural style transfer algorithm using VGG19.
"""

# Import necessary libraries
import torch
import argparse
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from utils import imload, imsave, contentLoss, gramLoss, tvLoss

# Define the main function to handle command line arguments and run experiments
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help="0 (noise) ~ 1 (content)")

    parser.add_argument('--content_image', type=str,
                        default="./imgs/golden_gate.jpg",
                        help="Content Image Path")

    parser.add_argument('--style_image', type=str,
                        default="./imgs/starry_night.jpg",
                        help="Style Image Path")

    parser.add_argument('--content_loss_weight', type=float, default=1.0,
                        help="Content Loss Weight")

    parser.add_argument('--style_loss_weight', type=float, default=50.0,  
                        help="Style Loss Weight")

    parser.add_argument('--tv_loss_weight', type=float, default=0.5,
                        help="Total Variation Loss Weight")

    parser.add_argument('--iteration', type=int, default=1000,
                        help="Number of iterations")

    parser.add_argument('--imsize', type=int, default=256,
                        help="Image Size")
                        
    parser.add_argument('--save_freq', type=int, default=250,
                        help="Frequency to save intermediate results")
                        
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help="Learning rate for optimizer")

    parser.add_argument('--save_path', type=str, default='./outputs/',
                        help="Save dir path")
                        
    args = parser.parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    
    # Extract style image name for file naming
    style_name = Path(args.style_image).stem
    
    # Save arguments for reference
    results_dir = save_path / f"{style_name}_nr{args.noise_ratio}_lr{args.learning_rate}"
    results_dir.mkdir(exist_ok=True)
    
    # Define style and content nodes for VGG19
    styleNodes = ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']
    contentNodes = ['relu_4_2']

    return_nodes = {'1': 'relu_1_1',
                    '6': 'relu_2_1',
                    '11': 'relu_3_1',
                    '20': 'relu_4_1',
                    '22': 'relu_4_2',
                    '29': 'relu_5_1'}

    # Check if CUDA is available and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ## Print GPU information if available - checking system
    """
    if device.type == 'cuda':
        # Print GPU information
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    """
    
    # Load VGG model
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    vgg.eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    feature_extractor = create_feature_extractor(vgg, return_nodes=return_nodes)

    # Load images
    start_time = time.time()
    print("Loading images...")
    content_image = imload(path=args.content_image,
                          imsize=args.imsize).to(device)
    style_image = imload(path=args.style_image,
                        imsize=args.imsize).to(device)
    print(f"Images loaded in {time.time() - start_time:.2f} seconds")

    # Pre-compute content and style features (only need to do this once)
    with torch.no_grad():
        contentFeatures = feature_extractor(content_image)
        style_features = feature_extractor(style_image)

    # Input image
    inputImage = torch.rand_like(content_image).to(device)
    inputImage = (inputImage * (1 - args.noise_ratio)) \
        + (content_image.detach() * args.noise_ratio)
    inputImage.requires_grad_(True)

    # Create optimizer with specified learning rate
    optimizer = torch.optim.LBFGS([inputImage], lr=args.learning_rate)
    
    # For tracking losses
    losses = {
        'content_loss': [],
        'style_loss': [],
        'tv_loss': [],
        'total_loss': []
    }
    
    # Start optimization
    startTime = time.time()
    
    # LBFGS optimization loop
    for i in tqdm(range(args.iteration), desc='Stylization'):
        def closure():
            """Closure function for LBFGS."""
            optimizer.zero_grad()
            
            # Extract features from input image
            inputFeatures = feature_extractor(inputImage)
            
            # Calculate losses
            content_loss = contentLoss(inputFeatures,
                                            contentFeatures,
                                            contentNodes)
            style_loss = gramLoss(inputFeatures,
                                        style_features,
                                        styleNodes)
            tv_loss = tvLoss(inputImage)
            
            # Weighted total loss
            total_loss = content_loss * args.content_loss_weight \
                + style_loss * args.style_loss_weight \
                + tv_loss * args.tv_loss_weight
            
            # Store losses for analysis
            losses['content_loss'].append(content_loss.item())
            losses['style_loss'].append(style_loss.item())
            losses['tv_loss'].append(tv_loss.item())
            losses['total_loss'].append(total_loss.item())
            
            total_loss.backward()
            
            return total_loss
        
        # Run optimization step
        optimizer.step(closure)
        
        # Save intermediate result at specified frequency
        if (i + 1) % args.save_freq == 0 or i == 0:
            # Save within subdirectory
            imsave(inputImage.detach().clone(), results_dir / f"iter_{i+1}.png")
            
            # Also save in main directory with parameter-specific name
            param_str = f"{style_name}_nr{args.noise_ratio}_lr{args.learning_rate}"
            imsave(inputImage.detach().clone(), save_path / f"{param_str}_iter{i+1}.png")
            
            # Log current status
            print(f"\nIteration {i+1}/{args.iteration}:")
            print(f"Content Loss: {losses['content_loss'][-1]:.4f}")
            print(f"Style Loss: {losses['style_loss'][-1]:.4f}")
            print(f"TV Loss: {losses['tv_loss'][-1]:.4f}")
            print(f"Total Loss: {losses['total_loss'][-1]:.4f}")
    
    # Calculate total time
    total_time = time.time() - startTime
    print(f"Stylization completed in {total_time:.2f} seconds")
    
    # Save final image with parameter-specific name
    param_str = f"{style_name}_nr{args.noise_ratio}_lr{args.learning_rate}"
    final_img_path = save_path / f"{param_str}.png"
    imsave(inputImage.detach().clone(), final_img_path)
    
    # Also save within the results directory
    imsave(inputImage.detach().clone(), results_dir / "final.png")
    
    # Save loss history
    np.save(results_dir / "loss_history.npy", losses)
    
    # Create a simple report
    report = {
        'total_time': total_time,
        'iterations': args.iteration,
        'device': str(device),
        'style_name': style_name,
        'noise_ratio': args.noise_ratio,
        'learning_rate': args.learning_rate,
        'final_losses': {
            'content_loss': losses['content_loss'][-1],
            'style_loss': losses['style_loss'][-1],
            'tv_loss': losses['tv_loss'][-1],
            'total_loss': losses['total_loss'][-1]
        },
        'args': vars(args)
    }
    
    # Save report in results directory
    import json
    with open(results_dir / "report.json", 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Results saved to {results_dir}")
    print(f"Final image saved as {final_img_path}")

# Run the main function if this script is executed directly
if __name__ == '__main__':
    main()