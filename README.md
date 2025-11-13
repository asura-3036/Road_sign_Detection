# Traffic Sign Recognition using Deep CNN

## Overview
This project implements a deep convolutional neural network (CNN) for recognizing German traffic signs using the GTSRB dataset. The model is trained to classify 43 traffic sign types, achieving 96.17% validation accuracy. It is designed to support robust, real-time traffic sign recognition for autonomous driving applications.

## Features
- Architecture with 5 convolutional blocks, including batch normalization, max pooling, and dropout to prevent overfitting and improve generalization
- Comprehensive data preprocessing including normalization and augmentation (rotation, zoom, shift, shear) for robustness to real-world conditions
- Training with Adam optimizer and categorical cross-entropy loss over 25 epochs with batch size 64
- Extensive performance evaluation with confusion matrices, ROC curves, precision/recall, and per-class accuracy analysis

## Installation

1. Clone the repository:
git clone <repository-url>
cd <repository-directory>

2. Install dependencies:
pip install tensorflow numpy matplotlib graphviz


3. Place the trained model file (`.h5` format) and the dataset in appropriate folders.

## Usage

- Load the pretrained model for inference or further training.
- Use provided scripts to perform inference on test images.
- Visualize model architecture with the included visualization scripts.
- Evaluate model performance using provided metrics and plots.

## Model Architecture

- Input layer for 32x32 RGB images
- Five convolutional blocks with increasing filters (64, 128, 256)
- Each block includes convolution layers with ReLU, batch normalization, max pooling, and dropout layers
- Final dense layers with dropout, followed by softmax output over 43 classes

## Dataset

- German Traffic Sign Recognition Benchmark (GTSRB) dataset with 51,839 labeled images across 43 classes.
- Dataset split: training, validation, and test sets with standardized 32x32 pixel images.

## Performance

- Achieved 98.6% training accuracy and 96.17% validation accuracy.
- Test set accuracy lower due to dataset discrepancies; ongoing work to address this.
- Detailed performance reported through confusion matrices and ROC curve analysis.

## Limitations

- Lower performance under extreme weather, occlusion, and viewpoint distortion.
- Inference latency around 50ms per image on CPU; GPU recommended for real-time deployment.

## Future Work

- Incorporate transfer learning and ensemble models for improved accuracy
- Include weather effect augmentation and hard negative mining
- Optimize model size with quantization and pruning for embedded deployment
- Implement continuous learning and production monitoring for deployed systems

## Contact

contact: bcs_2023016@iiitm.ac.in

contact: bcs_2023027@iiitm.ac.in

