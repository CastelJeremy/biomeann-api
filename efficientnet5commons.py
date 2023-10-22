import torch
from torchvision import transforms

model_kaggle = torch.load('models/efficientnet5/efficientnet5_05D_pretrained_epoch8_kaggle.pt', map_location=torch.device('cpu'))
model_homemade = torch.load('models/efficientnet5/efficientnet5_05D_pretrained_epoch14_homemade.pt', map_location=torch.device('cpu'))

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image
