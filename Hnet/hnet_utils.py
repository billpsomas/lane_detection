import os
import cv2
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt

from Hnet.hnet_model import HNet

# Use GPU if available, else use CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def hnet_transformation(input_pts, transformation_coefficient, poly_fit_order: int = 3, device: str = 'cuda'):
    """
    :param input_pts: the points of the lane of a single image, shape: [k, 3] (k is the number of points)
    :param transformation_coefficient: the 6 params from the HNet, shape: [1, 6]
    :param poly_fit_order: the order of the polynomial
    :param device: the device to use
    :return
    - valid_pts_reshaped (torch.Tensor): the valid points of the lane, shape: [3, k]
    - H (torch.Tensor): the transformation matrix (H), shape: [3, 3]
    - preds_transformation_back (torch.Tensor): the predicted and back-projected points of the lane, shape: [3, k]
    - pts_projects_normalized: the projected and normalized points of the lane, shape: [3, k]
    """
    H = torch.zeros(3, 3, device=device)

    # assign the h_prediction to the H matrix
    H[0, 0] = transformation_coefficient[0]  # a
    H[0, 1] = transformation_coefficient[1]  # b
    H[0, 2] = transformation_coefficient[2]  # c
    H[1, 1] = transformation_coefficient[3]  # d
    H[1, 2] = transformation_coefficient[4]  # e
    H[2, 1] = transformation_coefficient[5]  # f
    H[-1, -1] = 1
    H = H.type(torch.FloatTensor).to(device)

    # 2. transform input_pts using H matrix
    pts_reshaped = input_pts.transpose(0, 1)

    # 3. filter invalid points
    valid_points_indices = torch.where(pts_reshaped[2, :] == 1.)[0]
    valid_pts_reshaped = pts_reshaped[:, valid_points_indices]

    # 4. compute polynomial fit of transformed input_pts
    valid_pts_reshaped = valid_pts_reshaped.type(torch.FloatTensor).to(device)
    pts_projects = torch.matmul(H, valid_pts_reshaped)
    X = pts_projects[0, :] / pts_projects[2, :]
    Y = pts_projects[1, :] / pts_projects[2, :]
    Y_One = torch.ones_like(Y)
    if poly_fit_order == 2:
        Y_stack = torch.stack([torch.pow(Y, 2), Y, Y_One], dim=1)
    elif poly_fit_order == 3:
        Y_stack = torch.stack(
            [torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], dim=1)
    else:
        raise ValueError('Unknown order', poly_fit_order)

    Y_stack_t = Y_stack.transpose(0, 1)
    YtY = torch.matmul(Y_stack_t, Y_stack)
    YtY_inv = torch.pinverse(YtY)
    YtY_inv_Yt = torch.matmul(YtY_inv, Y_stack.transpose(0, 1))
    w = torch.matmul(YtY_inv_Yt, X.unsqueeze(1))

    x_preds = torch.matmul(Y_stack, w)
    preds = torch.transpose(torch.stack([torch.squeeze(x_preds, -1) * pts_projects[2, :],
                                         Y * pts_projects[2, :], pts_projects[2, :]], dim=1), 0, 1)

    # 5. transform polynomial fit back using H matrix
    preds_transformation_back = torch.matmul(torch.inverse(H), preds)

    # extra returns to use
    pts_projects_normalized = pts_projects / pts_projects[2, :]

    return valid_pts_reshaped, H, preds_transformation_back, pts_projects_normalized


def hnet_transform_back_points_after_polyfit(image, hnet_model, list_lane_pts, poly_fit_order: int = 3):
    """
    Transform back the lanes points after polynomial fit
    :param image: the image to transform back the lanes points. type: tensor shape: [1, 3, H, W]
    :param hnet_model: the hnet model, after loaded the weights
    :param list_lane_pts: the list of the lanes points to transform back. type: list of tensors shape: [k, 3]
    :param poly_fit_order: the order of the polynomial
    :return: the list of the transformed back lanes points. type: list of tensors shape: [k, 3]
    """
    # inference
    transformation_coefficient = hnet_model(image)
    # just to squeeze batch size to 1
    transformation_coefficient = transformation_coefficient[0]

    # multiply coefficents to scale by 4 because lanenet image is 4 time the hnet image
    multiplier = torch.tensor(
        [1., 1., 4, 1., 4., 0.25], dtype=torch.float32, device=device)
    transformation_coefficient = transformation_coefficient * multiplier

    # transform back all the lanes points
    preds_transformation_back_list = []
    # get transformed lanes points
    for lane_pts in list_lane_pts:
        if len(lane_pts) > 0 and lane_pts.shape[1] == 2:
            # add 1 to each point for homogeneous coordinates
            lane_pts = torch.concatenate(
                (lane_pts, torch.ones(lane_pts.shape[0], 1)), dim=1)

        _, _, preds_transformation_back, _ = hnet_transformation(lane_pts,
                                                                 transformation_coefficient,
                                                                 poly_fit_order)
        # valid_pts_reshaped, _,  preds_transformation_back, pts_projects_normalized = hnet_transformation(lane_pts,
        #                                                          transformation_coefficient,
        #                                                          poly_fit_order)
        # def save_coords_to_csv(valid_pts_reshaped, preds_transformation_back, pts_projects_normalized, filepath):
        #     import pandas as pd
        #     valid_pts_reshaped = valid_pts_reshaped.transpose(0, 1).detach().cpu().numpy()
        #     preds_transformation_back = preds_transformation_back.transpose(0, 1).detach().cpu().numpy()
        #     pts_projects_normalized = pts_projects_normalized.transpose(0, 1).detach().cpu().numpy()

        #     df = pd.DataFrame({
        #         'valid_pts_reshaped': np.round(valid_pts_reshaped, 2).tolist(),
        #         'preds_transformation_back': np.round(preds_transformation_back, 2).tolist(),
        #         'pts_projects_normalized': np.round(pts_projects_normalized, 2).tolist(),
        #     })

        #     df.to_csv(filepath)

        preds_transformation_back_list.append(
            preds_transformation_back.transpose(0, 1))
        # file_path = f"./{len(preds_transformation_back_list)}coords.csv"
        # save_coords_to_csv(valid_pts_reshaped, preds_transformation_back, pts_projects_normalized, file_path)

    return preds_transformation_back_list


def run_hnet_and_fit_from_lanenet_cluster(cluster_result_for_hnet,
                                          loaded_hnet_model, image,
                                          poly_fit_order=3,
                                          take_average_lane_cluster_pts=False,
                                          device_to_use='cuda'):
    """
    Run the hnet model and fit the lanes points from the lanenet cluster
    :param cluster_result_for_hnet: the cluster result from the lanenet model
    :param loaded_hnet_model: the loaded hnet model
    :param image: the image to run the hnet model on
    :param poly_fit_order: the order of the polynomial to fit
    :param take_average_lane_cluster_pts: if to take the average of the lane cluster points
    (same as in evaluation (see evaluate.py: "row_result")
    :param device_to_use: the device to use
    """
    image_hnet = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
    elements = np.unique(cluster_result_for_hnet)
    lanes_pts = []
    for line_idx in elements:
        if line_idx == 0:  # ignore background
            continue
        idx = np.where(cluster_result_for_hnet == line_idx)
        x, y = idx[1], idx[0]
        if take_average_lane_cluster_pts:
            # for every row (y) take the average of the cols (x) values, 
            # same as in evaluation (see evaluate.py: "row_result")
            x = np.array([np.mean(x[np.where(y == i)]) for i in np.unique(y)])
            y = np.unique(y)
        coord = np.vstack((x, y)).transpose()
        lanes_pts.append(coord)

        # draw the points on the src image
        # plt.figure()
        # plt.title(f"cluster {line_idx}")
        # plt.imshow(cv2.resize(image, (cluster_result_from_lanenet.shape[1], cluster_result_from_lanenet.shape[0]), interpolation=cv2.INTER_LINEAR))
        # plt.scatter(x, y, s=1, color='red')
        # plt.show()


    # transform list of numpy to list of torch
    lanes_pts = [torch.tensor(lane_pts) for lane_pts in lanes_pts]
    image_for_hnet_inference = torch.tensor(
        image_hnet, dtype=torch.float32, device=device_to_use)
    image_for_hnet_inference = image_for_hnet_inference.permute(2, 0, 1)
    # image_for_hnet_inference = torch.transpose(image_for_hnet_inference, (2, 0, 1))
    image_for_hnet_inference = image_for_hnet_inference.unsqueeze(0)
    # repeat so I have 10 batch
    image_for_hnet_inference = image_for_hnet_inference.repeat(10, 1, 1,
                                                               1)  # todo fix this so it doesn't have to be repeat as batch size
    lanes_transformed_back = hnet_transform_back_points_after_polyfit(image_for_hnet_inference, loaded_hnet_model,
                                                                      lanes_pts, poly_fit_order=poly_fit_order)
    # create mask in size of the image (128, 64) from the lanes
    fit_lanes_cluster_results = np.zeros(
        (cluster_result_for_hnet.shape[0], cluster_result_for_hnet.shape[1]), dtype=np.uint8)
    for i, lane in enumerate(lanes_transformed_back):
        for point in lane:           
            col_coord = np.clip(int(torch.round(point[0])), 0, cluster_result_for_hnet.shape[1]-1)
            row_coord = np.clip(int(torch.round(point[1])), 0, cluster_result_for_hnet.shape[0]-1)
            # +1 because the background is 0
            fit_lanes_cluster_results[row_coord, col_coord] = i + 1

    return image_hnet, lanes_transformed_back, fit_lanes_cluster_results


def draw_images(lane_points: torch.tensor, image: torch.tensor, transformation_coefficient,
                poly_fit_order, prefix_name, number, output_path):
    """
    Draw the lane points on the src image
    :param lane_points: the lane points of the src image (single image) (k, 3)
    :param image: the src image (3, H, W)
    :param transformation_coefficient: the transformation coefficient of the src image (6)
    :param poly_fit_order: the order of the polynomial to fit
    :param number: the number of the src image (index)
    :param prefix_name: prefix name for saving the images
    :param output_path: the output path of the images
    """
    valid_pts_reshaped, H, preds_transformation_back, pts_projects_normalized = hnet_transformation(
        input_pts=lane_points,
        transformation_coefficient=transformation_coefficient,
        poly_fit_order=poly_fit_order)
    src_image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

    # draw the points on the src image
    image_for_points = src_image.copy()
    points_for_drawing = valid_pts_reshaped.transpose(0, 1)
    for point in points_for_drawing:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image_for_points, center, 1, (0, 0, 255), -1)

    # draw the transformed back points on the src image
    image_for_transformed_back_points = src_image.copy()
    pred_transformation_back_for_drawing = preds_transformation_back.transpose(
        0, 1)
    for point in pred_transformation_back_for_drawing:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image_for_transformed_back_points,
                   center, 1, (0, 0, 255), -1)

    # draw the projected to bev image with lane
    # TODO maybe mid training in produce poor results?
    R = H.detach().cpu().numpy()
    pts_projects_normalized_for_drawing = pts_projects_normalized.transpose(
        0, 1)
    warp_image = cv2.warpPerspective(src_image, R, dsize=(
        src_image.shape[1], src_image.shape[0]))
    for point in pts_projects_normalized_for_drawing:
        center = (int(point[0]), int(point[1]))
        cv2.circle(warp_image, center, 1, (0, 0, 255), -1)

    # save the images
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(
        f"{output_path}/{prefix_name}_{number}_src.png", image_for_points)
    cv2.imwrite(
        f"{output_path}/{prefix_name}_{number}_transformed_back.png", image_for_transformed_back_points)
    cv2.imwrite(f"{output_path}/{prefix_name}_{number}_warp.png", warp_image)


def save_loss_to_pickle(loss_list: list, pickle_file_path: str = './pre_train_hnet_loss.pkl'):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(loss_list, f)


def plot_loss_from_pickle(pickle_file_path: str = './pre_train_hnet_loss.pkl'):
    with open(pickle_file_path, 'rb') as f:
        loss_list = pickle.load(f)
    plot_loss(loss_list)


def plot_loss(loss_list: list, title: str = 'Pretrain HNet Loss', output_path: str = None):
    # create new figure
    plt.figure()
    # plot so x-axis will start from 1 same as epochs
    plt.scatter(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid()
    if output_path:
        title_as_snake_case = title.lower().replace(' ', '_')
        output_path_with_extension = os.path.join(
            output_path, f"{title_as_snake_case}.png")
        plt.savefig(output_path_with_extension)
    # close the figure
    plt.close()


def save_hnet_model_with_info(hnet_model: HNet, phase, epoch, batch_size,
                              poly_order, regularization=False, output_path=None):
    """
    Save the hnet model with all the info needed to resume training
    :param hnet_model: the hnet model to save
    :param phase: the phase of the training: train or pre_train
    :param epoch: the epoch of the training
    :param batch_size: the batch size of the training
    :param poly_order: the polynomial order of the polynomial fit
    :param regularization: if regularization was used
    :param output_path: the output path to save the model
    """
    if output_path is None:
        raise ValueError('output_path cannot be None')
    # save model state along with other parameters
    torch.save({'state_dict': hnet_model.state_dict(),
                'phase': phase,
                'epoch': epoch,
                'batch_size': batch_size,
                'poly_order': poly_order,
                'regularization': regularization},
               output_path)
    

def load_hnet_model_with_info(hnet_model: HNet, model_path: str, device: str = 'cuda'):
    """
    Load the hnet model with all the info needed to resume training
    :param hnet_model: the hnet model to load the weights to
    :param model_path: the path of the model to load
    :param device: the device to use
    :return: the info dict
    """
    if device == 'cuda':
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint.keys():
        hnet_model.load_state_dict(checkpoint['state_dict'])
    else:
        # backward compatibility for back when only the weights were saved
        hnet_model.load_state_dict(checkpoint)
    info_dict_to_return = {}
    if 'phase' in checkpoint.keys():
        info_dict_to_return['phase'] = checkpoint['phase']
    if 'epoch' in checkpoint.keys():    
        info_dict_to_return['epoch'] = checkpoint['epoch']
    if 'batch_size' in checkpoint.keys():
        info_dict_to_return['batch_size'] = checkpoint['batch_size']
    if 'poly_order' in checkpoint.keys():
        info_dict_to_return['poly_order'] = checkpoint['poly_order']
    if 'regularization' in checkpoint.keys():
        info_dict_to_return['regularization'] = checkpoint['regularization']
    return info_dict_to_return

def get_x_threshes_of_gt_evaluation(gt, y_samples):
    """
    copy of the evalutation part to use in drawings
    """
    from utils.lane import LaneEval
    angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
    threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
    return threshs
            