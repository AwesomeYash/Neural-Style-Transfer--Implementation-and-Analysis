# Neural Style Transfer using VGG19  
**Name:** Priyanshu Ranka (NUID: 002035396)  
**Professor:** Prof. Bruce Maxwell  
**Course:** CS 5330 - Pattern Recognition and Computer Vision  
**Semester:** Spring 2025 

This project implements a PyTorch-based Neural Style Transfer algorithm based on the approach introduced by Gatys et al. The implementation allows for systematic experimentation with different hyperparameters to analyze their effects on stylization quality.

## Project Links
- **Code Files:** [https://northeastern-my.sharepoint.com/:f:/g/personal/ranka_pr_northeastern_edu/EqywOadPP4RFkvkPk7wLTk4BHRzkHYn6GTZ4oKn82abkbA?e=QHnx0a]
- **Images:** [https://northeastern-my.sharepoint.com/:f:/g/personal/ranka_pr_northeastern_edu/Eu-dIClJ9shEm0qo9Qs8oLIBo05Tmf3W1uSISM1jO9QXBg?e=pKD9eM]
- **Outputs:** [https://northeastern-my.sharepoint.com/:f:/g/personal/ranka_pr_northeastern_edu/EumQEPT9f7xOpbYX0k0eJ9IByUB7AwPh6fbNLuK1IGAHzA?e=Fz2EFj]


## Project Structure
```
Submission Files
│
├── Code Files/                  # Source code for the implementation
│   ├── main.py                  # Core implementation of neural style transfer
│   ├── loss.py                  # Loss function definitions for content, style, and TV
│   ├── utils.py                 # Helper functions for image processing and visualization
│   ├── run_parameters.py        # Script for automating parameter experiments
│   └── analysis.py              # Tools for analyzing loss behavior and convergence
│
├── imgs/                        # Source images directory
│   ├── content/                 # Content images
│   └── style/                   # Style reference images
│
└── outputs/                     # Generated outputs and results
    ├── {style}_nr{noise_ratio}_lr{learning_rate}/   # Individual experiment results
    │   ├── final.png            # Final stylized image
    │   ├── iter_{n}.png         # Intermediate results at specified iterations
    │   ├── loss_history.npy     # Recorded loss values
    │   └── report.json          # Experiment parameters and metrics
    └── parameter_comparison.png # Visual comparison of different parameter settings
```

## Requirements
- Python 3.6+
- PyTorch 1.8+
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)
- tqdm

You can install the required packages using:

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

## Usage
### Basic Style Transfer
To perform style transfer with default parameters:

```bash
python main.py --content_image ./imgs/your_content.jpg --style_image ./imgs/your_style.jpg
```

### Advanced Options
The implementation supports various parameters for fine-tuning the style transfer process:

```bash
python main.py --content_image ./imgs/your_content.jpg \
               --style_image ./imgs/your_style.jpg \
               --noise_ratio 0.5 \
               --learning_rate 0.7 \
               --content_loss_weight 1.0 \
               --style_loss_weight 10.0 \
               --tv_loss_weight 0.5 \
               --iteration 1000 \
               --imsize 256 \
               --save_freq 250 \
               --save_path ./outputs/
```

### Parameter Description
- `--noise_ratio`: Controls the initialization of the input image (0.0 = noise, 1.0 = content image)
- `--learning_rate`: Learning rate for the L-BFGS optimizer
- `--content_loss_weight`: Weight for content loss component
- `--style_loss_weight`: Weight for style loss component
- `--tv_loss_weight`: Weight for total variation regularization
- `--iteration`: Number of optimization iterations
- `--imsize`: Size of the images during processing
- `--save_freq`: Frequency for saving intermediate results
- `--save_path`: Directory to save result images

### Running with Multiple Parameter Combinations
To experiment with different parameter combinations:

```bash
python run_parameters.py --content_image ./imgs/your_content.jpg \
                        --style_image ./imgs/your_style.jpg \
                        --iterations 1000 \
                        --imsize 256 \
                        --output_dir ./outputs
```

This will run neural style transfer with various combinations of noise ratios and learning rates, saving the results in separate folders.

### Analyzing Results
To analyze loss convergence and generate visualizations:

```bash
python analysis.py --loss_file ./outputs/{style}_nr{noise_ratio}_lr{learning_rate}/loss_history.npy \
                  --output_dir ./analysis
```

## Implementation Details
### Feature Extraction
The implementation uses the pre-trained VGG19 network as a feature extractor, following the approach described by Gatys et al. Content representation is captured from layer `relu4_2`, while style representation incorporates multiple layers (`relu1_1` through `relu5_1`).

### Loss Functions
The implementation incorporates three key loss components:

1. **Content Loss**: Mean squared error between content feature representations
2. **Style Loss**: Mean squared error between Gram matrices of style features
3. **Total Variation Loss**: Regularization term to ensure spatial smoothness

### Optimization Process
The optimization employs L-BFGS with a closure function for gradient updates. The process iteratively refines the input image to minimize the weighted combination of content, style, and total variation losses.

## Parameter Effects
### Noise Ratio
- **Higher noise ratios (0.7+)**: Preserve more content structure but may result in less pronounced stylization
- **Medium noise ratios (0.4-0.6)**: Balance content preservation and style adoption
- **Lower noise ratios (0.3-)**: Produce more creative stylizations but may compromise content structure

### Learning Rate
- **Higher learning rates (0.8+)**: Achieve faster initial stylization but may introduce artifacts
- **Medium learning rates (0.5-0.7)**: Balance exploration and refinement of style elements
- **Lower learning rates (0.4-)**: Produce smoother, more refined results but require more iterations

## Results
The outputs directory contains styled images with filenames indicating the parameters used. For each parameter combination, the following files are generated:

- `{style}_nr{noise_ratio}_lr{learning_rate}.png`: Final stylized image
- `{style}_nr{noise_ratio}_lr{learning_rate}/iter_{iteration}.png`: Intermediate results
- `{style}_nr{noise_ratio}_lr{learning_rate}/loss_history.npy`: Loss values during optimization
- `{style}_nr{noise_ratio}_lr{learning_rate}/report.json`: Summary of the run parameters and results

## Acknowledgments
This implementation is based on the following papers:

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution.
- Mahendran, A., & Vedaldi, A. (2015). Understanding deep image representations by inverting them.
