import os
import os.path as ops
import numpy as np
import torch
import cv2
import sys
import pdb
import matplotlib.pyplot as plt

from dataset_utils import TUSIMPLE, TUSIMPLE_AUG

# Directory of train, valid, test .txt files 
root = '../TUSIMPLE/txt_for_local'

# Build train, valid, test set
train_set = TUSIMPLE(root=root, flag='train')
valid_set = TUSIMPLE(root=root, flag='valid')
test_set = TUSIMPLE(root=root, flag='test')

# Print the number of samples of each set
print('train_set length {}'.format(len(train_set)))
print('valid_set length {}'.format(len(valid_set)))
print('test_set length {}'.format(len(test_set)))

# Let's visualize some sample images 
# First, change directory, so that the images can be loaded from TUSIMPLE/
os.chdir('../')

# Choice a random index to get the corresponding
# sample image, binary lane image, instance lane image
idx = 120
gt, bgt, igt = train_set[idx]

# Print some information about those images
print('image type {}'.format(type(gt)))
print('image size {} \n'.format(gt.size()))

print('gt binary image type {}'.format(type(bgt)))
print('gt binary image size {}'.format(bgt.size()))
print('items in gt binary image {} \n'.format(torch.unique(bgt)))

print('gt instance type {}'.format(type(igt)))
print('gt instance size {}'.format(igt.size()))
print('items in gt instance {} \n'.format(torch.unique(igt)))

# Make a folder to save sample images
try:
    os.makedirs('dataset/samples')
except:
    FileExistsError

# Change again directory to /dataset
os.chdir('dataset')

# Show/save sample images
sample_image = ((gt.numpy() + 1) * 127.5).astype(int)
sample_image.shape

plt.figure(figsize=(15,15))

sample_image = sample_image.transpose(1,2,0)
sample_image = sample_image[...,::-1]

plt.imsave('samples/sample_image.png', np.uint8(sample_image))
#plt.imshow(sample_image)
#plt.show()

# Shape of the corresponding binary image
bgt.shape

# Show/save corresponding binary image
plt.figure(figsize=(20,20))
ax1 = plt.subplot(121)
#plt.imshow(bgt, cmap='gray')
plt.imsave('samples/binary_sample_image.png', np.uint8(bgt), cmap='gray')

# Show/save corresponding instance image (as grayscale)
ax1 = plt.subplot(122)
#plt.imshow(igt, cmap='gray')
plt.imsave('samples/instance_sample_image.png', np.uint8(igt), cmap='inferno')

# Let's make some augmentations now
# Build augmented train, valid, test set
train_set = TUSIMPLE_AUG(root=root, flag='train')
valid_set = TUSIMPLE_AUG(root=root, flag='valid')
test_set = TUSIMPLE_AUG(root=root, flag='test')

# Print the number of samples of each set
print('train_set length {}'.format(len(train_set)))
print('valid_set length {}'.format(len(valid_set)))
print('test_set length {}'.format(len(test_set)))

# Change again directory, so that the images can be loaded from TUSIMPLE/
os.chdir('../')

# Choice again a random index to get the corresponding
# sample image, binary image, instance image, as also the corresponding
# sample augmented image, augmented binary image, augmented instance image
idx = 120
gt, bgt, igt = train_set[idx]
gt_aug, bgt_aug, igt_aug = train_set[idx+1]

# Print some information about those images
print('image type {}'.format(type(gt)))
print('image size {} \n'.format(gt.size()))

print('gt binary image type {}'.format(type(bgt)))
print('gt binary image size {}'.format(bgt.size()))
print('items in gt binary image {} \n'.format(torch.unique(bgt)))

print('gt instance type {}'.format(type(igt)))
print('gt instance size {}'.format(igt.size()))
print('items in gt instance {} \n'.format(torch.unique(igt)))

# Change again directory to /dataset
os.chdir('dataset')

# Show/save sample images & sample augmented images
original_sample_image = ((gt.numpy() + 1) * 127.5).astype(int)
aug_sample_image = ((gt_aug.numpy() + 1) * 127.5).astype(int)
original_sample_image.shape

plt.figure(figsize=(20,20))

# Show/save original sample image
ax1 = plt.subplot(121)
original_sample_image = original_sample_image.transpose(1,2,0)
original_sample_image = original_sample_image[...,::-1]
#plt.imshow(original_sample_image)
plt.imsave('samples/original_sample_image.png', np.uint8(original_sample_image), cmap='gray')

# Show/save augmented sample image
ax1 = plt.subplot(122)
aug_sample_image = aug_sample_image.transpose(1,2,0)
aug_sample_image = aug_sample_image[...,::-1]
# plt.imshow(aug_sample_image)
plt.imsave('samples/aug_sample_image.png', np.uint8(aug_sample_image), cmap='gray')

# Show/save corresponding original binary image
plt.figure(figsize=(20,20))
ax1 = plt.subplot(121)
#plt.imshow(bgt, cmap='gray')
plt.imsave('samples/original_binary_image.png', np.uint8(bgt), cmap='gray')

# Show/save corresponding original instance image (as grayscale)
ax1 = plt.subplot(122)
#plt.imshow(igt, cmap='gray')
plt.imsave('samples/original_instance_image.png', np.uint8(igt), cmap='inferno')

# Show/save corresponding augmented binary image
plt.figure(figsize=(20,20))
ax1 = plt.subplot(121)
#plt.imshow(bgt_aug, cmap='gray')
plt.imsave('samples/aug_binary_image.png', np.uint8(bgt_aug), cmap='gray')

# Show/save corresponding original instance image (as grayscale)
ax1 = plt.subplot(122)
plt.imshow(igt_aug, cmap='gray')
plt.imsave('samples/aug_instance_image.png', np.uint8(igt_aug), cmap='inferno')