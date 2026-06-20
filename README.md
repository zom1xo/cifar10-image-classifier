# CIFAR-10 Image Classifier

Transfer learning with pretrained ResNet18 for CIFAR-10 image classification. Built during Internspark AI internship.

## Overview
- **Dataset:** CIFAR-10 (60,000 32×32 images, 10 classes).
- **Model:** ResNet18 pretrained on ImageNet, modified for CIFAR-10 input size.
- **Training:** 10 epochs, data augmentation, StepLR scheduler, GPU (T4).

## Results
| Metric | Value |
|--------|-------|
| Train Accuracy | 96.79% |
| Val Accuracy | 91.08% |
| Test Accuracy | 91.77% |

## Top Confusions
- Cat ↔ Dog: 187
- Car ↔ Truck: 50
- Bird ↔ Cat: 46

## Files
- `Task2_CIFAR10_Classifier.ipynb` — Full training notebook.
- `cifar10_resnet18.pth` — Saved model weights (Google Drive).
- `INFERENCE.md` — Instructions to load and run the saved model.

## Tools
PyTorch, torchvision, Scikit-learn, Matplotlib, Seaborn, Google Colab

## Author
[Alan Raju](https://www.linkedin.com/in/alan-raju-09bb0040a/) | [GitHub](https://github.com/zom1xo)
