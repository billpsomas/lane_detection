import os
import cv2
import time
import glob
import torch
import pickle
import argparse
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt

import hnet_data_processor
from Hnet.hnet_loss import HNetLoss
from hnet_model import HNet
from Hnet.hnet_utils import hnet_transformation
from pre_train_hnet_loss import PreTrainHnetLoss


PRE_TRAIN_LEARNING_RATE = 1e-4
TRAIN_LEARNING_RATE = 5e-5
WEIGTH_DECAY = 0.0002


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_dir', type=str, help='The origin path of unzipped tusimple train dataset',
                        default='~/Downloads/train_set')
    parser.add_argument('--batch_size', type=int, help='The batch size of the dataset',
                        default=10)
    parser.add_argument('--phase', type=str,
                        help='The phase is train, pretrain or full_train', default='pretrain')

    # pre train phase arguments
    parser.add_argument('--pre_train_epochs', type=int,
                        help='The pre train epochs', default=5)
    parser.add_argument('--pre_hnet_weights', type=str,
                        help='The pre hnet weights path', required=False)
    parser.add_argument('--pre_train_save_dir', type=str, help='The pre train hnet weights save dir',
                        default='./pre_train_hnet_weights')

    # train phase arguments
    parser.add_argument('--train_epochs', type=int,
                        help='The train epochs', default=5)
    parser.add_argument('--hnet_weights', type=str,
                        help='The hnet model weights path', required=False)
    parser.add_argument('--train_save_dir', type=str, help='The train hnet weights save dir',
                        default='./train_hnet_weights')
    return parser.parse_args()


def train(args):
    # Define the batch size
    batch_size = args.batch_size

    # Build train set
    train_set = hnet_data_processor.TusimpleForHnetDataSet(
        set_directory=args.data_set_dir, flag='train'
    )
    print("train_set length {}".format(len(train_set)))

    # Define DataLoaders
    data_loader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Use GPU if available, else use CPU
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define the model
    hnet_model = HNet()
    hnet_model.to(device)

    # # Define scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.1)
    #
    # # Define number of epochs
    # num_epochs = 20005

    assert args.phase in ['pretrain', 'train', 'full_train'], "phase must be pretrain, train or full_train"

    if args.phase == 'pretrain':
        pre_train_hnet(args, data_loader_train, device, hnet_model)
    elif args.phase == 'train':
        train_hnet(args, data_loader_train, device, hnet_model)
    else:
        pre_train_hnet(args, data_loader_train, device, hnet_model)
        train_hnet(args, data_loader_train, device, hnet_model)


def train_hnet(args, data_loader_train, device, hnet_model):
    # Define the optimizer
    params = [p for p in hnet_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=TRAIN_LEARNING_RATE, weight_decay=WEIGTH_DECAY)
    epochs = args.train_epochs

    if args.hnet_weights is not None:
        hnet_model.load_state_dict(torch.load(args.hnet_weights))
        print("Load train hnet weights success")
    else:
        print("No train hnet weights")
    train_loss = HNetLoss()

    epochs_loss = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        curr_epoch_loss_list = []
        for i, (gt_images, gt_lane_points) in enumerate(data_loader_train):
            gt_images = Variable(gt_images).to(device).type(torch.float32)
            gt_lane_points = Variable(gt_lane_points).to(device)

            # todo: filter bad input points:
            # something like this: points = lane_points[lane_points[:, 2] > 0]

            optimizer.zero_grad()
            transformation_coefficient = hnet_model(gt_images)
            loss = train_loss(gt_lane_points, transformation_coefficient)
            loss.backward()
            optimizer.step()

            curr_epoch_loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}s'
                      .format(epoch, epochs, i + 1, len(data_loader_train), loss.item(),
                              time.time() - start_time))
                start_time = time.time()

            # draw_images(gt_lane_points[0], gt_images[0], transformation_coefficient[0], i)

        epochs_loss.append(np.mean(curr_epoch_loss_list))

        if (epoch + 1) % 1 == 0:
            os.makedirs(args.pre_train_save_dir, exist_ok=True)
            file_path = os.path.join(
                args.pre_train_save_dir, 'pre_train_hnet_epoch_{}.pth'.format(epoch))
            torch.save(hnet_model.state_dict(), file_path)
    # save loss list to a pickle file
    save_loss_to_pickle(epochs_loss)


def pre_train_hnet(args, data_loader_train, device, hnet_model):
    # Define the optimizer
    params = [p for p in hnet_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=PRE_TRAIN_LEARNING_RATE, weight_decay=WEIGTH_DECAY)
    epochs = args.pre_train_epochs

    if args.pre_hnet_weights is not None:
        hnet_model.load_state_dict(torch.load(args.pre_hnet_weights))
        print("Load pretrain hnet weights success")
    else:
        print("No pretrain hnet weights")
    pre_train_loss = PreTrainHnetLoss()

    epochs_loss = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        curr_epoch_loss_list = []
        for i, (gt_images, gt_lane_points) in enumerate(data_loader_train):
            gt_images = Variable(gt_images).to(device).type(torch.float32)
            gt_lane_points = Variable(gt_lane_points).to(device)

            optimizer.zero_grad()
            transformation_coefficient = hnet_model(gt_images)
            loss = pre_train_loss(transformation_coefficient)
            loss.backward()
            optimizer.step()

            curr_epoch_loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.4f}s'
                      .format(epoch, epochs, i + 1, len(data_loader_train), loss.item(),
                              time.time() - start_time))
                start_time = time.time()

            draw_images(gt_lane_points[0], gt_images[0], transformation_coefficient[0], i)

        epochs_loss.append(np.mean(curr_epoch_loss_list))

        if (epoch + 1) % 1 == 0:
            os.makedirs(args.pre_train_save_dir, exist_ok=True)
            file_path = os.path.join(
                args.pre_train_save_dir, 'pre_train_hnet_epoch_{}.pth'.format(epoch))
            torch.save(hnet_model.state_dict(), file_path)
    # save loss list to a pickle file
    save_loss_to_pickle(epochs_loss)


def save_loss_to_pickle(loss_list: list, pickle_file_path: str = './pre_train_hnet_loss.pkl'):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(loss_list, f)


def plot_loss(loss_pickle_file_pah: str = './pre_train_hnet_loss.pkl'):
    with open(loss_pickle_file_pah, 'rb') as f:
        loss_list = pickle.load(f)
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretrain HNet Loss')
    plt.grid()
    plt.show()


def draw_images(lane_points, image, transformation_coefficient, number):
    """
    Draw the lane points on the src image
    :param lane_points: the lane points of the src image (single image)
    :param image: the src image
    :param transformation_coefficient: the transformation coefficient of the src image (single image)
    :param number: the number of the src image (index)
    """
    points = lane_points[lane_points[:, 2] > 0]
    x_transformation_back, H, pts_projects_nomalized = hnet_transformation(
        input_pts=points.unsqueeze(dim=0), transformation_coefficient=transformation_coefficient.unsqueeze(dim=0))
    # draw gt lane on src image 
    src_image = image.transpose(1, 2, 0).numpy()
    for i in range(points.shape[1]):
        cv2.circle(src_image, (int(points[0, i]), int(points[1, i])), 5, (0, 0, 255), -1)
    cv2.imwrite(f"pre_train_hnet_save_number_{number}_src.png", src_image)


if __name__ == '__main__':
    # plot_loss()
    args = parse_args()
    train(args)
