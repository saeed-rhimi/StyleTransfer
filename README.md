
# Neural Style Transfer with PyTorch

This repository contains an implementation of neural style transfer using PyTorch and the VGG19 model. The objective of neural style transfer is to blend the content of one image with the style of another image, creating a unique and visually appealing result.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Overview](#code-overview)
  - [Importing Required Libraries](#importing-required-libraries)
  - [Load Pretrained VGG19 Model](#load-pretrained-vgg19-model)
  - [Set Up Device for GPU Usage](#set-up-device-for-gpu-usage)
  - [Load Content and Style Images](#load-content-and-style-images)
  - [Preprocess Images](#preprocess-images)
  - [Display Initial Images](#display-initial-images)
  - [Define Feature Extraction and Gram Matrix Functions](#define-feature-extraction-and-gram-matrix-functions)
  - [Extract and Display Feature Maps](#extract-and-display-feature-maps)
  - [Perform Style Transfer](#perform-style-transfer)
  - [Display Final Images](#display-final-images)
- [License](#license)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)
- [Useful Links](#useful-links)

## Installation

To use this code, you need to have Python and PyTorch installed. You can install the required libraries using the following command:

```bash
pip install torch torchvision imageio numpy matplotlib
```

## Usage

1. Place your content image as `input2.jpg` and style image as `style.jpg` in the root directory.
2. Run the script to perform style transfer and generate the output image.

## Project Structure

- `style_transfer.py`: Main script for neural style transfer.
- `input2.jpg`: Content image.
- `style.jpg`: Style image.

## Code Overview

### Importing Required Libraries

The code begins by importing all necessary libraries such as PyTorch for deep learning, torchvision for computer vision tasks, imageio for reading images, numpy for numerical operations, and matplotlib for visualization. Additionally, warnings are suppressed for a cleaner output.

### Load Pretrained VGG19 Model

The VGG19 model is loaded with pretrained weights. All parameters are frozen to prevent them from being updated during the style transfer process. The model is set to evaluation mode to disable layers like dropout and batch normalization.

### Set Up Device for GPU Usage

The code checks if a GPU is available and sets the computation device accordingly. The model is then moved to the selected device to utilize GPU acceleration if available.

### Load Content and Style Images

The content and style images are loaded from the file system. A target image is also created, initialized with random pixel values but with the same shape as the content image.

### Preprocess Images

The images are preprocessed using a series of transformations: converting to tensor, resizing to 256x256 pixels, and normalizing using the mean and standard deviation values typical for ImageNet-trained models.

### Display Initial Images

The initial content, style, and target images are displayed side by side using matplotlib to provide a visual reference before the style transfer process begins.

### Define Feature Extraction and Gram Matrix Functions

Two essential functions are defined:
1. `extract_features`: Extracts feature maps from specific layers of the VGG19 model.
2. `compute_gram_matrix`: Computes the Gram matrix, which captures style information by calculating the correlations between feature maps.

### Extract and Display Feature Maps

Feature maps and Gram matrices for the content and style images are extracted and displayed for a subset of layers. This visualization helps in understanding how different layers capture various aspects of the images.

### Perform Style Transfer

The core of the style transfer process involves optimizing the target image to minimize the content loss (difference from content image features) and style loss (difference from style image Gram matrices). An optimization loop is run for a specified number of epochs, updating the target image iteratively.

### Display Final Images

After the optimization loop, the final content, target, and style images are displayed to show the result of the style transfer process. The target image should now blend the content of the content image with the style of the style image.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact Information

For any inquiries or questions, please contact the project maintainer:

- Name: [Your Name]
- Email: [your.email@example.com]
- GitHub: [Your GitHub Profile](https://github.com/yourusername)

## Acknowledgments

Special thanks to the following individuals and resources for their contributions and support:

- The PyTorch team for their extensive documentation and resources.
- The developers of the VGG19 model for providing the pretrained weights.
- [Any other individuals or resources you would like to acknowledge]

## Useful Links

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Documentation](https://pytorch.org/vision/stable/index.html)
- [Imageio Documentation](https://imageio.readthedocs.io/en/stable/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Neural Style Transfer Papers and Tutorials](https://www.google.com/search?q=neural+style+transfer+papers+and+tutorials)

Feel free to explore and modify the code to see how different parameters affect the output. Happy experimenting!
