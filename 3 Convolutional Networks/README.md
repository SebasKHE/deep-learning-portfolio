# Module 3: Convolutional Neural Networks

This module showcases advanced computer vision projects using state-of-the-art CNN architectures for real-world applications.

## 📂 Projects

### 1. CNN Fundamentals
**File**: `Convolution_model_Step_by_Step_v1.ipynb`

Building convolutional neural networks from scratch.

**Implementation**:
- Convolutional layers with multiple filters
- Pooling layers (max and average)
- Forward and backward propagation for CNNs
- Complete CNN architecture assembly

---

### 2. CNN Application - Sign Language Recognition
**File**: `Convolution_model_Application.ipynb`

Custom CNN for recognizing hand sign language gestures.

**Achievements**:
- Designed and trained multi-layer CNN
- Applied data preprocessing and augmentation
- Achieved high accuracy on sign language classification
- Demonstrated end-to-end CNN workflow

---

### 3. Residual Networks (ResNet)
**File**: `Residual_Networks.ipynb`

Implementation of deep residual networks with skip connections.

**Technical Highlights**:
- Built identity and convolutional blocks
- Implemented skip connections to enable very deep networks
- Demonstrated solution to vanishing gradient problem
- Trained 50+ layer networks successfully

---

### 4. Transfer Learning with MobileNet
**File**: `Transfer_learning_with_MobileNet_v1.ipynb`

Fine-tuning pre-trained MobileNet for custom image classification.

**Transfer Learning Workflow**:
- Loaded pre-trained MobileNet weights
- Froze base layers and added custom classification head
- Fine-tuned on custom dataset
- Achieved excellent results with limited training data

---

### 5. Object Detection - Autonomous Driving
**File**: `Autonomous_driving_application_Car_detection.ipynb`

YOLO-based car detection system for autonomous vehicles.

**Implementation**:
- Applied YOLO (You Only Look Once) architecture
- Implemented bounding box predictions
- Non-max suppression for overlapping detections
- Real-time object detection capabilities

---

### 6. Face Recognition System
**File**: `Face_Recognition.ipynb`

Siamese network for face verification and recognition.

**Technical Approach**:
- Implemented triplet loss function
- Built face encoding system
- Face verification (1:1 comparison)
- Face recognition (1:K identification)
- Applied to security and authentication scenarios

---

### 7. Image Segmentation with U-Net
**File**: `Image_segmentation_Unet_v2.ipynb`

Semantic segmentation for medical imaging using U-Net architecture.

**Key Features**:
- Encoder-decoder architecture with skip connections
- Pixel-wise classification
- Medical image segmentation
- Demonstrated precision in boundary detection

---

### 8. Neural Style Transfer
**File**: `Art_Generation_with_Neural_Style_Transfer.ipynb`

Artistic style transfer using VGG network.

**Creative Application**:
- Extracted content and style representations from VGG
- Optimized generated image to match both content and style
- Implemented Gram matrix for style representation
- Created artistic renditions of photographs

## 🎯 Skills Demonstrated

- **CNN Architectures**: Custom CNNs, ResNet, MobileNet, U-Net, YOLO
- **Computer Vision Tasks**: Classification, object detection, segmentation, style transfer
- **Transfer Learning**: Fine-tuning pre-trained models
- **Advanced Techniques**: Skip connections, triplet loss, non-max suppression
- **Real-World Applications**: Autonomous driving, face recognition, medical imaging
- **Framework Mastery**: TensorFlow, Keras for complex architectures

## 🚀 Running the Notebooks

Each notebook is self-contained:

```bash
jupyter notebook "Residual_Networks.ipynb"
```

Note: Some projects may require downloading pre-trained weights or datasets. Instructions are provided within each notebook.

---

[← Back to Main Repository](../README.md)
