'''
This is for kaggle dataset
Penguins vs Turtles
Image classification with bounding boxes
https://www.kaggle.com/datasets/abbymorgan/penguins-vs-turtles?datasetId=3202424&sortBy=dateRun&tab=profile

You must edit it fit on your dataset.
'''

import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

train_length, valid_length = 500, 72

# Load annotations
train_labels = json.loads(open('../data/train/annotations').read())
valid_labels = json.loads(open('../data/valid/annotations').read())

train_labels = [0 if item["category_id"] == 2 else 1 for item in train_labels]
valid_labels = [0 if item["category_id"] == 2 else 1 for item in valid_labels]

# Image normalization values
mean = [0.6012, 0.6324, 0.6453]
std = [0.2472, 0.2300, 0.2304]

torch.manual_seed(42)
    
class PenguinVSTurtleDataset(Dataset):
    """
    Dataset class for the Penguin vs Turtle classification task.

    Args:
        length (int): Length of the dataset.
        transform (callable): Transform to be applied on the images.
        annotations (list): List of image annotations.
        train (bool): Flag indicating if the dataset is for training or validation.
    """
    def __init__(self, length, annotations, train=True):
        super().__init__()
        self.len = length
        self.annotations = annotations
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), # Image resizing: 224 x 224 x 3
            transforms.RandomHorizontalFlip(), # Data augmentation: Random horizontal flip
            transforms.ToTensor(),
            transforms.Normalize(mean, std) # Normalization
        ]) if train else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, id):
        path = f'../data/train/images/{id}.jpg' if self.train else f'../data/valid/images/{id}.jpg'
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = train_labels[id] if self.train else valid_labels[id]

        return (image, label)

train_set = PenguinVSTurtleDataset(train_length, train_labels, train=True)
valid_set = PenguinVSTurtleDataset(valid_length, valid_labels, train=False)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=2)

if __name__ == "__main__":
    for images, labels in train_loader:
        assert images.shape == (16, 3, 224, 224) and labels.shape == (16,), "Incorrect shape for images or labels."
        break
    for images, labels in valid_loader:
        assert images.shape == (16, 3, 224, 224) and labels.shape == (16,), "Incorrect shape for images or labels."
        break
    print("Dataset and DataLoader are working correctly.")