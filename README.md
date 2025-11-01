# ML-CV-Projects

A collection of machine learning and computer vision projects built while learning PyTorch and production ML practices.

## Projects

### 1. CIFAR-10 Image Classifier

**CNN built from scratch** | 75% accuracy

- Custom CNN architecture (268K parameters)
- Trained on 50,000 images
- [View Project →](./cifar10_classifier/)

### 2. Transfer Learning Flower Classifier

**ResNet18 Transfer Learning** | 84% accuracy

- Pre-trained ImageNet weights
- 5-class classification (3,670 images)
- Production features: argparse, GPU training, visualization
- [View Project →](./transfer_learning_flowers/)

### 1. ResNet Flowers-102 Classifier 🌸 **[NEW]**

**Production ML | Transfer Learning | Web Deployment** | 81.49% accuracy

- ResNet18 with class imbalance handling
- 102-class flower species classification
- **🚀 [Live Demo](https://resnetflagship-8jatnx6d22xdntk63jmapx.streamlit.app/)**
- Production features: mixed precision, early stopping, data augmentation
- Deployed web app with Streamlit Cloud
- [View Project →](./resnet_flagship/)

## Tech Stack

**Training:** PyTorch • CUDA • torchvision • Albumentations • YAML  
**Deployment:** Streamlit • Streamlit Cloud  
**Tools:** matplotlib • NumPy • Pillow • Python
