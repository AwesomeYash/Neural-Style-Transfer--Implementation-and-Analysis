"""
Neural Style Transfer using VGG19

Name: Priyanshu Ranka (NUID: 002035396)
Professor: Prof. Bruce Maxwell
Course: CS 5330 - Pattern Recognition and Computer Vision
Semester: Spring 2025
Description: Analyzes the loss history of the neural style transfer algorithm.
"""
# Import necessary libraries
import argparse
from pathlib import Path
from utils import plot_losses

# Define the main function to handle command line arguments and run experiments
def main():
    parser = argparse.ArgumentParser(description='Analyze style transfer loss history')
    parser.add_argument('--loss_file', type=str, required=True, 
                        help='Path to the loss history .npy file')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot losses
    plot_losses(args.loss_file, output_dir)
    
# Main entry point
if __name__ == '__main__':
    main()