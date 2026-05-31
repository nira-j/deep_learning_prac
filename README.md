# Colored Bubble Detection from OMR Sheets using ResNet50

## Overview

This project implements a deep learning-based Optical Mark Recognition (OMR) system that detects and classifies colored bubbles on OMR sheets using a Convolutional Neural Network (CNN) built on the ResNet50 architecture.

Traditional OMR systems rely on image processing techniques and thresholding methods, which can struggle with varying lighting conditions, scanning quality, and colored markings. This project leverages transfer learning with ResNet50 to achieve robust and accurate bubble classification.

---

## Features

*  Automatic detection of colored bubbles from OMR sheets
*  ResNet50-based CNN architecture
*  Transfer learning for improved accuracy
*  Supports multiple bubble colors
*  Used SVM(Support Vector Mchine) for classification among classes.

---

## Dataset

The dataset consists of cropped bubble images extracted from OMR sheets.

### Classes

Example classes:

* Option A filled
* Option B filled
* Option C filled
* Option D filled
* Any two option filled
* Blank no option filled

Dataset Structure:

```text
dataset/
├── train/
│   ├── A/
│   ├── B/
│   ├── C/
│   └── D/
|   └── astric/
|   └── blank/
|
└── test/
|   ├── A/
│   ├── B/
│   ├── C/
│   └── D/
|   └── astric/
|   └── blank/
```

---

## Model Architecture

The model uses ResNet50 as a feature extractor.

```text
Input Image
      ↓
ResNet50 (Pretrained on ImageNet)
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Dropout
      ↓
Softmax Output Layer
```

### Why ResNet50?

* Deep architecture with residual connections
* Strong feature extraction capabilities
* Faster convergence using transfer learning
* High accuracy on image classification tasks

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
