import os

# Rename training images
for f in range(0, 500):
    os.rename(f'./train/images/image_id_{f:03d}.jpg', f'./train/{f}.jpg')

# Rename validation images
for f in range(0, 72):
    os.rename(f'./valid/images/image_id_{f:03d}.jpg', f'./valid/{f}.jpg')