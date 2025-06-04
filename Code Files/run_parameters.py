"""
Neural Style Transfer using VGG19

Name: Priyanshu Ranka (NUID: 002035396)
Professor: Prof. Bruce Maxwell
Course: CS 5330 - Pattern Recognition and Computer Vision
Semester: Spring 2025
Description: This script runs multiple style transfer experiments with different parameters.

"""

# Import necessary libraries
import subprocess
import argparse
from pathlib import Path
import time

# Define the main function to handle command line arguments and run experiments
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run style transfer with multiple parameters")
    parser.add_argument('--content_image', type=str, required=True,
                        help="Path to content image")
    parser.add_argument('--style_image', type=str, required=True,
                        help="Path to style image")
    parser.add_argument('--iterations', type=int, default=1000,
                        help="Number of iterations for each run")
    parser.add_argument('--imsize', type=int, default=256,
                        help="Image size for processing")
    parser.add_argument('--save_freq', type=int, default=250,
                        help="Frequency to save intermediate results")
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help="Base directory to save all results")
    args = parser.parse_args()
    
    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define parameter combinations to test
    noise_ratios = [0.3, 0.5, 0.7]
    learning_rates = [0.4, 0.7, 1.0]
    
    # Get total number of combinations
    total_runs = len(noise_ratios) * len(learning_rates)
    run_count = 1
    
    # Total time tracking
    start_time = time.time()
    
    # Run all combinations
    for nr in noise_ratios:
        for lr in learning_rates:
            print(f"\n{'='*80}")
            print(f"Run {run_count}/{total_runs}: noise_ratio={nr}, learning_rate={lr}")
            print(f"{'='*80}")
            
            # Build command
            cmd = [
                "python", "main.py",
                "--content_image", args.content_image,
                "--style_image", args.style_image,
                "--noise_ratio", str(nr),
                "--learning_rate", str(lr),
                "--iteration", str(args.iterations),
                "--imsize", str(args.imsize),
                "--save_freq", str(args.save_freq),
                "--save_path", str(output_dir)
            ]
            
            # Run the command
            try:
                subprocess.run(cmd, check=True)
                print(f"Completed run {run_count}/{total_runs}")
            except subprocess.CalledProcessError as e:
                print(f"Error in run {run_count}: {e}")
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                break
                
            run_count += 1
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print total time taken and completion message
    print(f"\nAll runs completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to {output_dir}")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()