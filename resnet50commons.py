import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

class CustomResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, weights=None):
        super(CustomResNet, self).__init__()
        self.model = models.resnet50(weights=weights) 
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
        
class CustomResNetNoDrop(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(CustomResNetNoDrop, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_features = self.model.fc.in_features
        out_features = self.model.fc.out_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

import __main__
setattr(__main__, "CustomResNet", CustomResNet)
setattr(__main__, "CustomResNetNoDrop", CustomResNetNoDrop)

model_homemade = torch.load('models/resnet50/resnet50_05D_pretrained_epoch19_homemade.pt', map_location=torch.device('cpu'))
model_kaggle = torch.load('models/resnet50/resnet50_0D_pretrained_epochES_kaggle.pt', map_location=torch.device('cpu'))

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image
