import os.path as ops
import numpy as np
import torch
import cv2
import time
import os
import matplotlib.pylab as plt
import sys
from tqdm import tqdm
import imageio
from dataset.dataset_utils import TUSIMPLE
from Lanenet.model import Lanenet
from utils.evaluation import gray_to_rgb_emb, process_instance_embedding, video_to_clips

# Load the model 
model_path = 'TUSIMPLE/Lanenet_output/lanenet_epoch_39_batch_8.model'

# Initialize model and send it to cpu for visualization
LaneNet_model = Lanenet(2, 4)
LaneNet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Load the test set
root = 'TUSIMPLE/txt_for_local'
test_set = TUSIMPLE(root=root, flag='test')

# Print the length of test set
print('test_set length {}'.format(len(test_set)))

# Choice a random index to get the corresponding
# sample image, binary lane image, instance lane image from test set
idx = 0
gt, bgt, igt = test_set[idx]

# Define the corresponding DataLoader
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

# Make a forward of sample image to obtain the output from the network
binary_final_logits, instance_embedding = LaneNet_model(gt.unsqueeze(0))

# Print the shapes of the outputs
print('binary_final_logits shape: {}'.format(binary_final_logits.shape))
print('instance_embedding shape: {}'.format(instance_embedding.shape))

# Show/save ground truths
# gt is the sample image
# bgt is the binary segmentation ground truth image
# igt is the instance segmentation ground truth image
sample_image = ((gt.numpy() + 1) * 127.5).astype(int)
sample_image.shape

# Make a folder to save ground truth images
# in order to compare them with predictions
try:
    os.makedirs('TUSIMPLE/test_clips/groundtruths')
except:
    FileExistsError

# Show/save sample image
plt.figure(figsize=(20,10))

sample_image = sample_image.transpose(1,2,0)
sample_image = sample_image[...,::-1]

plt.imsave('TUSIMPLE/test_clips/groundtruths/sample_image.png', np.uint8(sample_image))
#plt.imshow(sample_image)
#plt.show()

# Shape of the corresponding binary image
bgt.shape

# Show/save corresponding binary image
plt.figure(figsize=(20,20))
ax1 = plt.subplot(121)
#plt.imshow(bgt, cmap='gray')
plt.imsave('TUSIMPLE/test_clips/groundtruths/binary_groundtruth_sample_image.png', np.uint8(bgt), cmap='gray')

# Show/save corresponding instance image (as grayscale)
ax1 = plt.subplot(122)
#plt.imshow(igt, cmap='gray')
plt.imsave('TUSIMPLE/test_clips/groundtruths/instance_groundtruth_sample_image.png', np.uint8(igt), cmap='inferno')

# Show/save network predictions
# binary_img is the binary segmentation prediction image
# instance_embedding is the instance segmentation embedding
# which is further processed by the clustering algorith in order
# to obtain the instance segmenation prediction image rgb_emb
binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
rgb_emb, cluster_result = process_instance_embedding(instance_embedding, binary_img, distance=1, lane_num=5)

# Show sample image (it is already previously saved in groundtruths/ folder)
#plt.figure(figsize=(20,10))
#ax1 = plt.subplot(221)
#plt.imshow(sample_image)
#plt.title('Original sample image')

# Make a folder to save prediction images
# in order to compare them with groundtruths
try:
    os.makedirs('TUSIMPLE/test_clips/predictions')
except:
    FileExistsError

# Show/save binary lane segmentation prediction on sample image 
#ax1 = plt.subplot(222)
#plt.imshow(binary_img, cmap='gray')
#plt.title('Binary Lane Segmentation Prediction')
plt.imsave('TUSIMPLE/test_clips/predictions/binary_prediction_sample_image.png', binary_img, cmap='gray')

# Show/save instance segmentation prediction on sample image 
#ax1 = plt.subplot(223)
#plt.imshow(rgb_emb)
#plt.title('Instance Segmentation Prdiction')
plt.imsave('TUSIMPLE/test_clips/predictions/instance_prediction_sample_image.png', rgb_emb, cmap='inferno')

# Show/save the final result with the lane detection on sample image
#ax1 = plt.subplot(224)
a = 0.7
#plt.imshow(a*sample_image/255 + (1-a)*rbg_emb)
#plt.title('Final Detection Result')
plt.imsave('TUSIMPLE/test_clips/predictions/final_detection_sample_image.png', a*sample_image/255 + (1-a)*rgb_emb, cmap='inferno')


