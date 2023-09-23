import os
import cv2
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt

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
        Y_stack = torch.stack([torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], dim=1)
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
        output_path_with_extension = os.path.join(output_path, f"{title_as_snake_case}.png")
        plt.savefig(output_path_with_extension)
    # close the figure
    plt.close()


def draw_images(lane_points: torch.tensor, image: torch.tensor, transformation_coefficient,
                prefix_name, number, output_path):
    """
    Draw the lane points on the src image
    :param lane_points: the lane points of the src image (single image) (k, 3)
    :param image: the src image (3, H, W)
    :param transformation_coefficient: the transformation coefficient of the src image (6)
    :param number: the number of the src image (index)
    :param prefix_name: prefix name for saving the images
    :param output_path: the output path of the images
    """
    valid_pts_reshaped, H, preds_transformation_back, pts_projects_normalized = hnet_transformation(
        input_pts=lane_points,
        transformation_coefficient=transformation_coefficient)
    src_image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

    # draw the points on the src image
    image_for_points = src_image.copy()
    points_for_drawing = valid_pts_reshaped.transpose(0, 1)
    for point in points_for_drawing:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image_for_points, center, 1, (0, 0, 255), -1)

    # draw the transformed back points on the src image
    image_for_transformed_back_points = src_image.copy()
    pred_transformation_back_for_drawing = preds_transformation_back.transpose(0, 1)
    for point in pred_transformation_back_for_drawing:
        center = (int(point[0]), int(point[1]))
        cv2.circle(image_for_transformed_back_points,
                   center, 1, (0, 0, 255), -1)

    # draw the projected to bev image with lane
    # TODO maybe mid training in produce poor results?
    R = H.detach().cpu().numpy()
    pts_projects_normalized_for_drawing = pts_projects_normalized.transpose(0, 1)
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