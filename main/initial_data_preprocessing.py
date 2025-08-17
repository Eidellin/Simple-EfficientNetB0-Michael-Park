import os

"""
You may not use it.

move 'train_annotations' file to 'train' folder and rename to 'annotations'
move 'valid_annotations' file to 'valid' folder and rename to 'annotations'
move images in 'train' folder to 'train/images' folder
move images in 'valid' folder to 'valid/images' folder
"""

# Create a folder named "images" in both train and valid directories
if not os.path.exists('../data/train/images'):
    os.makedirs('../data/train/images')
if not os.path.exists('../data/valid/images'):
    os.makedirs('../data/valid/images')

# Move and rename annotation files
os.rename('../data/train_annotations', '../data/train/annotations')
os.rename('../data/valid_annotations', '../data/valid/annotations')

# Rename training images
for f in range(0, 500):
    os.rename(f'../data/train/image_id_{f:03d}.jpg', f'../data/train/images/{f}.jpg')

# Rename validation images
for f in range(0, 72):
    os.rename(f'../data/valid/image_id_{f:03d}.jpg', f'../data/valid/images/{f}.jpg')