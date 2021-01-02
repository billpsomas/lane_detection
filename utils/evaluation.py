import os.path as ops
import numpy as np
import torch
import cv2
import time
import tqdm
import os
from sklearn.cluster import MeanShift, estimate_bandwidth


def gray_to_rgb_emb(gray_img):
    """
    :param gray_img: torch tensor 256 x 512
    :return: numpy array 256 x 512
    """
    H, W = gray_img.shape
    element = torch.unique(gray_img).numpy()
    rbg_emb = np.zeros((H, W, 3))
    color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 215, 0], [0, 255, 255]]
    for i in range(len(element)):
        rbg_emb[gray_img == element[i]] = color[i]
    return rbg_emb/255


def process_instance_embedding(instance_embedding, binary_img, distance=1, lane_num=5):
    # The embedding of each pixel of image
    # Let us mention that it's a 4-dimensional embedding
    embedding = instance_embedding[0].detach().numpy().transpose(1, 2, 0)
    # Initialize a zero matrix having the image's shape
    cluster_result = np.zeros(binary_img.shape, dtype=np.int32)
    # binary_img>0 is True for those pixels that correspond to a lane
    # By this way we create a list having the embedding values only for those pixels
    # that correspond to a lane
    cluster_list = embedding[binary_img > 0]
    # Define and run MeanShift algorithm on previous embeddings
    mean_shift = MeanShift(bandwidth=distance, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(cluster_list)
    # The labels occured from MeanShift for every pixel
    # Let us remind the fact that this label corresponds
    # to a different lane
    labels = mean_shift.labels_
    # Write the labels in the zero matrix previously defined
    # Let us remind this matrix has image's shape
    cluster_result[binary_img > 0] = labels + 1
    # If MeanShift found more clusters than the 
    # number we wanted then assign those pixels with the background value
    cluster_result[cluster_result > lane_num] = 0
    
    # Also if MeanShift found clusters with less than 15 pixels
    # then assign those pixels with the background value
    for idx in np.unique(cluster_result):
        if len(cluster_result[cluster_result == idx]) < 15:
            cluster_result[cluster_result == idx] = 0

    # Let's visualize our result
    H, W = binary_img.shape
    rgb_emb = np.zeros((H, W, 3))
    color = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 215, 0], [0, 255, 255]]
    element = np.unique(cluster_result)
    
    for i in range(len(element)):
        rgb_emb[cluster_result == element[i]] = color[i]

    return rgb_emb / 255, cluster_result


def video_to_clips(video_name):
    test_video_dir = ops.split(video_name)[0]
    outimg_dir = ops.join(test_video_dir, 'clips')
    if ops.exists(outimg_dir):
        print('Data already exist in {}'.format(outimg_dir))
        return
    if not ops.exists(outimg_dir):
        os.makedirs(outimg_dir)
    video_cap = cv2.VideoCapture(video_name)
    frame_count = 0
    all_frames = []

    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1

    for i, frame in enumerate(all_frames):
        out_frame_name = '{:s}.png'.format('{:d}'.format(i + 1).zfill(6))
        out_frame_path = ops.join(outimg_dir, out_frame_name)
        cv2.imwrite(out_frame_path, frame)
    print('finish process and save in {}'.format(outimg_dir))




