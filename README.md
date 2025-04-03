# Ninjacart-CV-Classification

**Overview**

Ninjacart is India's largest fresh produce supply chain company, leveraging technology to solve one of the toughest supply chain challenges—delivering fresh products from farmers to businesses within 12 hours. This case study focuses on developing a robust image classification model to distinguish between different types of vegetables and filter out noisy (irrelevant) images.

**Dataset
**
- The dataset consists of 3,135 images categorized into four folders, representing different vegetable types and noise.
- Each image is preprocessed through resizing, normalization, and augmentation to improve model performance.

**Problem Statement**

The objective is to build a deep learning-based classifier that accurately classifies vegetable images while distinguishing irrelevant images (noise) to ensure correct product delivery in Ninjacart's supply chain.

**Approach**

**Data Preprocessing:**

- Images were resized to a fixed dimension.
- Normalization was applied to scale pixel values.
- Augmentation techniques such as rotation, flipping, and brightness adjustments were used.

**Model Selection & Training:** 

- Implemented a Convolutional Neural Network (CNN) for classification.
- Transfer learning approaches using ResNet and EfficientNet were explored.
- Hyperparameter tuning was performed to optimize batch size, learning rate, and number of epochs.

**Performance Evaluation:**

- Accuracy, Precision, Recall, and F1-score were used as evaluation metrics.
- A confusion matrix revealed misclassifications, particularly among visually similar vegetables.

**Results**

- The CNN model achieved good classification accuracy with some misclassifications.
- Transfer learning models like ResNet and EfficientNet improved overall accuracy.
- The model had difficulty distinguishing between certain vegetable categories and noisy images in some cases.

**Recommendations**

- Enhance data augmentation with additional techniques such as zoom, contrast adjustments, and elastic transformations.
- Address class imbalance using SMOTE or class-weighted loss functions.
- Experiment with deeper CNN architectures and hyperparameter tuning (Grid Search, Bayesian Optimization).
- Deploy the model in a real-world environment using TensorFlow Lite or ONNX for efficient inference.
