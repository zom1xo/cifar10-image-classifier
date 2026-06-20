# How to Load and Run the CIFAR-10 Model

The trained model weights are stored on Google Drive. Download `cifar10_resnet18.pth` from the link below, then follow these steps.

## Download Model
https://drive.google.com/file/d/1ZsA8iMEi4NEhdCJ1VkEeBg1yWsxzNhCs/view?usp=sharing

## Requirements
- Python 3.x
- PyTorch, torchvision, Pillow

## Load and Predict

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Rebuild architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)

# 2. Load weights
model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device))
model.eval()

# 3. Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

# 4. Predict
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return classes[predicted.item()]

# Example:
# print(predict('my_cat_photo.jpg'))
