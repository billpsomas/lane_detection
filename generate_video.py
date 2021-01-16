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
idx = 22
gt, bgt, igt = test_set[idx]

# Define the corresponding DataLoader
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

# Make a forward of sample image to obtain the output from the network
binary_final_logits, instance_embedding = LaneNet_model(gt.unsqueeze(0))

# Print the shapes of the outputs
print('binary_final_logits shape: {}'.format(binary_final_logits.shape))
print('instance_embedding shape: {}'.format(instance_embedding.shape))

# Generate video for visualization
def clips_to_gif(test_clips_root, git_root):
    img_paths = []
    for img_name in os.listdir(test_clips_root):
        img_paths.append(ops.join(test_clips_root,img_name))
    img_paths.sort()
    gif_frames = []
    for i, img_name in enumerate(img_paths):
        gt_img_org = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))

        binary_final_logits, instance_embedding = LaneNet_model(gt_image.unsqueeze(0))
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
        binary_img[0:65,:] = 0
        rbg_emb, cluster_result = process_instance_embedding(instance_embedding, binary_img,
                                                             distance=1.5, lane_num=4)

        rbg_emb = cv2.resize(rbg_emb, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)
        a = 0.6
        frame = a * gt_img_org[..., ::-1] / 255 + rbg_emb * (1 - a)
        frame = np.rint(frame * 255)
        frame = frame.astype(np.uint8)
        gif_frames.append(frame)
    imageio.mimsave(git_root, gif_frames, fps=5)

clips_root = 'TUSIMPLE/test_clips'
gif_dir = 'TUSIMPLE/gif_output'
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)
for dir_name in os.listdir(clips_root):
    if dir_name == '.DS_Store' or dir_name == 'groundtruths' or dir_name == 'predictions':
        continue
    print('Process the clip {} \n'.format(dir_name))
    test_clips_root = ops.join(clips_root, dir_name)
    git_root = ops.join(gif_dir, dir_name) + '.gif'
    clips_to_gif(test_clips_root, git_root)