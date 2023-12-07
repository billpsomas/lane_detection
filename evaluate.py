import copy
import json
import os.path as ops
import numpy as np
import torch
import cv2
import time
import os
import matplotlib.pylab as plt
import sys

from tqdm import tqdm
from dataset.dataset_utils import TUSIMPLE
from Lanenet.model import Lanenet
from utils.evaluation import process_instance_embedding, video_to_clips

import argparse
from Hnet.hnet_model import HNet
from Hnet.hnet_utils import load_hnet_model_with_info
from Hnet.hnet_utils import run_hnet_and_fit_from_lanenet_cluster

# Use GPU if available, else use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lanenet_model_path', type=str, help='lanenet model path', required=True)
    parser.add_argument('--hnet_model_path', type=str, help='hnet model path', required=True)
    parser.add_argument('--poly_order', type=int, 
                        help='poly order to fit when evaultating. if info exist in loaded hnet model, use loaded value', 
                        required=False, default=2)

    return parser.parse_args()

def evaluate(args):

    ##### TODO old remove later #####
    # Load the model you want to evaluate
    # You can use either the simple or augmented
    # Uncomment corresponding line in second case
    model_path = 'TUSIMPLE/Lanenet_output/lanenet_epoch_39_batch_8.model'
    # model_path = 'TUSIMPLE/Lanenet_output/lanenet_epoch_39_batch_8_AUG.model'
    model_path = "/home/tal/git/University/Lane_Detection_PyTorch_project/TUSIMPLE/Lanenet_output/lanenet_epoch_29_batch_size_8.model"

    # hnet_model_path = "/home/tal/git/University/Lane_Detection_PyTorch_project/train/weights/train_hnet_poly_order_2_epoch_20.pth"
    hnet_model_path = "train/weights/train_hnet_poly_order_2_epoch_5_from_other_pc.pth"
    ##### old remove later #####

    model_path = args.lanenet_model_path
    hnet_model_path = args.hnet_model_path


    # Initialize model and load its parameters
    LaneNet_model = Lanenet(2, 4)
    LaneNet_model.load_state_dict(torch.load(model_path))
    LaneNet_model.to(device)
    print('Lanenet Model successfully loaded!')

    # Initialize the Hnet model and load its parameters
    hnet_model = HNet()
    # hnet_model.load_state_dict(torch.load(hnet_model_path))
    loaded_hnet_info_dict = load_hnet_model_with_info(hnet_model, hnet_model_path)
    hnet_model.to(device)
    poly_order = loaded_hnet_info_dict.get('poly_order', args.poly_order)
    print('Hnet Model successfully loaded!')

    # Read the test_tasks_0627.json file
    # This is the file in which we will append the lanes detected
    pred_json_path = 'TUSIMPLE/test_set/test_tasks_0627.json'
    json_pred = [json.loads(line) for line in open(pred_json_path).readlines()]

    all_time_forward = []
    all_time_clustering = []
    all_time_hnet_and_fit = []

    for i, sample in enumerate(tqdm(json_pred)):
        # Define variables corresponding to the fields of json file
        h_samples = sample['h_samples']
        lanes = sample['lanes']
        run_time = sample['run_time']
        raw_file = sample['raw_file']
        img_path = ops.join('TUSIMPLE/test_set', raw_file)

        # Read image and preprocess it in order to be ready 
        # to go through the network 
        gt_img_org = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))

        # Pass image through the network
        time_start = time.time()

        # Use GPU if available, else use CPU
        if torch.cuda.is_available():
            binary_final_logits, instance_embedding = LaneNet_model(gt_image.unsqueeze(0).cuda())
        else:
            binary_final_logits = binary_final_logits.cpu()
            instance_embedding = instance_embedding.cpu()

        time_end = time.time()

        # Get the final embedding
        # This is the binary image of the binary segmentation 
        # of lanes produced by the network
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().cpu().numpy()
        binary_img[0:50, :] = 0

        # Clustering phase
        # Let us remind that in this phase the binary segmentation image 
        # along with the instance embedding is fed into the clustering MeanShift algorithm 
        # and the algorithm outputs a number of clusters
        # each one of those corresponds to a different lane
        clu_start = time.time()
        rgb_emb, cluster_result = process_instance_embedding(instance_embedding.cpu(), binary_img, distance=1.5, lane_num=4)
        clu_end = time.time()

        cluster_result_for_hnet = copy.deepcopy(cluster_result)

        cluster_result = cv2.resize(cluster_result, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_NEAREST)
        elements = np.unique(cluster_result)

        # Hnet fit phase
        run_hnet_fit_start = time.time()
        # transform the lanes points back from the lanenet clusters
        image_hnet, lanes_transformed_back, fit_lanes_cluster_results = run_hnet_and_fit_from_lanenet_cluster(
            cluster_result_for_hnet, hnet_model, gt_img_org, poly_fit_order=poly_order)
        
        run_hnet_fit_end = time.time()
        # resize lanes mask to original size
        fit_lanes_cluster_results = cv2.resize(fit_lanes_cluster_results,
                                            dsize=(org_shape[1], org_shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
        # sanity check - if there are the same amount of valid lane points in each lane cluster its wierd
        number_of_different_valid_points_in_each_lane = len(
            np.unique(np.unique(fit_lanes_cluster_results, return_counts=True)[1][1:]))
        if number_of_different_valid_points_in_each_lane < 4:
            print("wierd, should be 4 but number_of_different_valid_points_in_each_lane = {}".format(
                number_of_different_valid_points_in_each_lane))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(gt_img_org, cv2.COLOR_BGR2RGB))
        plt.imshow(fit_lanes_cluster_results, interpolation='nearest', alpha=0.3, cmap='inferno')
        # add side plot with cluster_results mask over the gt image
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(gt_img_org, cv2.COLOR_BGR2RGB), cmap='inferno')
        plt.imshow(cluster_result, interpolation='nearest', alpha=0.3, cmap='inferno')

        plt.title('Mask with 5 Labels')
        plt.show()

        for line_idx in elements:
            if line_idx == 0:
                continue
            else:
                mask = (cluster_result == line_idx)
                fit_lanes_mask = (fit_lanes_cluster_results == line_idx)

                select_mask = mask[h_samples]
                select_mask = fit_lanes_mask[h_samples] # todo - use this to evaluate the model after hnet poly fit
                row_result = []
                for row in range(len(h_samples)):
                    col_indexes = np.nonzero(select_mask[row])[0]
                    if len(col_indexes) == 0:
                        row_result.append(-2)
                    else:
                        row_result.append(int(col_indexes.min() + (col_indexes.max() - col_indexes.min()) / 2))

                # Append results to .json file
                json_pred[i]['lanes'].append(row_result)
                json_pred[i]['run_time'] = time_end - time_start
                all_time_forward.append(time_end - time_start)
                all_time_clustering.append(clu_end - clu_start)
                all_time_hnet_and_fit.append(run_hnet_fit_end - run_hnet_fit_start)

    # Calculate the average duration 
    # of a forward pass as also of a clustering run
    forward_avg = np.sum(all_time_forward[500:2000]) / 1500
    cluster_avg = np.sum(all_time_clustering[500:2000]) / 1500

    # Print the durations
    print('The forward pass time for one image is: {}ms'.format(forward_avg * 1000))
    print('The clustering time for one image is: {}ms'.format(cluster_avg * 1000))
    print('The total time for one image is: {}ms'.format((cluster_avg + forward_avg) * 1000))

    print('The speed for forward pass is: {}fps'.format(1 / forward_avg))
    print('The speed for forward pass is: {}fps'.format(1 / cluster_avg))

    # Write the results in the pred.json file
    # or in the pred_aug.json file in case you used
    # the model trained on augmented data
    # Uncomment corresponding line in second case
    with open('TUSIMPLE/pred.json', 'w') as f:
        # with open('TUSIMPLE/pred_aug.json', 'w') as f:
        for res in json_pred:
            json.dump(res, f)
            f.write('\n')

    # Evaluate our results using the TUSIMPLE competition script 
    # and their groundtruths
    from utils.lane import LaneEval

    # Uncomment corresponding line in case 
    # you used the model trained on augmented data
    result = LaneEval.bench_one_submit('TUSIMPLE/pred.json', 'TUSIMPLE/test_set/test_label.json')
    # result = LaneEval.bench_one_submit('TUSIMPLE/pred_aug.json','TUSIMPLE/test_set/test_label.json')

    print(result)


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)