'''
This is for kaggle dataset
Penguins vs Turtles
Image classification with bounding boxes
https://www.kaggle.com/datasets/abbymorgan/penguins-vs-turtles?datasetId=3202424&sortBy=dateRun&tab=profile
'''

import os
import re
import sys
import json
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
    
class PenguinVSTurtleDataset(Dataset):
    def __init__(self, paths, train_labels, valid_labels, transform):
        super().__init__()
        self.paths = paths
        self.len = len(self.paths)
        self.transform = transform
        self.type = type
        self.train_labels = train_labels
        self.valid_labels = valid_labels

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = int(re.findall(r'\d+', path)[-1])
        if path.find('train'):
            label = self.train_labels[label]
        else:
            label = self.valid_labels[label]
        return (image, label)
    
# Function for Penguins vs Turtles
def load_data_for_Penguins_vs_Turtles(archive_path='../data'):
    train = f'{archive_path}/train/'
    valid = f'{archive_path}/valid/'

    def train_path(p): return f"{train}/{p}"
    train = list(map(train_path, os.listdir(train)))
    def valid_path(p): return f"{valid}/{p}"
    valid = list(map(valid_path, os.listdir(valid)))

    try:
        ratio = int(sys.argv[1])
        if ratio > 0 and ratio < 10:
            ratio /= 10
            data = train + valid
            random.shuffle(data)
            train = data[:int(len(data)*ratio)]
            valid = data[int(len(data)*ratio):]
            del data
    except:
        pass

    train_annotations = json.loads(open(f'{archive_path}/train_annotations').read())
    valid_annotations = json.loads(open(f'{archive_path}/valid_annotations').read())

    train_labels = [item["category_id"] for item in train_annotations]
    valid_labels = [item["category_id"] for item in valid_annotations]
    
    train_labels = [0 if label == 1 else 1 for label in train_labels]
    valid_labels = [0 if label == 1 else 1 for label in valid_labels]

    train_annotations = json.loads(open(f'{archive_path}/train_annotations').read())
    valid_annotations = json.loads(open(f'{archive_path}/valid_annotations').read())

    train_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.60012704, 0.63211549, 0.6454381], [0.30825471, 0.28704487, 0.29295797])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.60012704, 0.63211549, 0.6454381], [0.30825471, 0.28704487, 0.29295797])
    ])
    
    train_set = PenguinVSTurtleDataset(train, train_labels, valid_labels, train_transform)
    valid_set= PenguinVSTurtleDataset(valid, train_labels, valid_labels, valid_transform)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)
    
    return train_loader, valid_loader