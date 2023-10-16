# DenseNet, GoogleNet, and ResNet50 for Image Classification

This repository showcases the implementation of three powerful deep learning architectures: DenseNet121, GoogleNet (Inception V1), and ResNet50, for image classification tasks. These models are renowned for their effectiveness in various computer vision applications.

## Models

### DenseNet121
- DenseNet121 is a densely connected convolutional network that efficiently utilizes feature maps through dense blocks and transition layers. It enables the training of deep networks with fewer parameters.

### GoogleNet (Inception V1)
- GoogleNet, also known as Inception V1, introduced the concept of inception modules. Inception modules perform parallel convolutions with different kernel sizes and concatenate the results. This design captures features at various scales.

### ResNet50
- ResNet50 employs residual connections, allowing the model to learn residual functions. It includes shortcut connections, addressing the vanishing gradient problem and enabling the training of very deep networks.

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/saidislombek-abdusamatov/network_architectures.git
   ```

2. **Install Dependencies:**

   ```bash
   pip install tensorflow
   ```

3. **Import Model**

   ```python
   from archeitectures.DenseNet121 import DenseNet121
   from archeitectures.ResNet50 import ResNet50
   from archeitectures.GoogleNet import GoogleNet
   ```
