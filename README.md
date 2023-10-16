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
   pip install tensorflow scikit-learn
   ```

3. **Train the Models:**

   - To train DenseNet121, run `train_densenet.py`.
   - To train GoogleNet, run `train_googlenet.py`.
   - To train ResNet50, run `train_resnet50.py`.
   
   Each script loads the dataset, preprocesses the images, builds and trains the respective model. The trained models will be saved for future predictions.

4. **Evaluate the Models:**

   - After training, you can evaluate the models using test data or new images. Modify the corresponding `predict_*.py` scripts to load the trained models and make predictions on new images.

## Citation

```
@InProceedings{Nilsback08,
   author = "Yang, Yi and Newsam, Shawn",
   title = "Bag-Of-Visual-Words and Spatial Extensions for Image Classification",
   booktitle = "ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS)",
   year = "2010",
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to modify and use this README template for your own projects on GitHub!
