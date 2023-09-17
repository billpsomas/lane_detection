import os
import cv2
import glob
import json
import torch
import numpy as np


class TusimpleForHnetDataSet(torch.utils.data.Dataset):
    def __init__(self, set_directory, resize=(128, 64), flag='train'):

        self.resize = resize

        if flag == 'train':
            # take all json files under the train directory as list
            dataset_info_files = glob.glob('{:s}/*.json'.format(set_directory))
        else:
            raise NotImplementedError('Not implemented yet')

        self._label_image_path, self._label_gt_pts = self._init_data_set_by_json_files(
            dataset_info_files)

    @staticmethod
    def _init_data_set_by_json_files(dataset_info_files):

        label_image_path = []
        label_gt_pts = []

        for json_file_path in dataset_info_files:
            assert os.path.exists(
                json_file_path), '{:s} not exist'.format(json_file_path)

            src_dir = os.path.split(json_file_path)[0]

            with open(json_file_path, 'r') as file:
                for line in file:
                    info_dict = json.loads(line)

                    image_dir = os.path.split(info_dict['raw_file'])[0]
                    image_dir_split = image_dir.split('/')[1:]
                    image_dir_split.append(
                        os.path.split(info_dict['raw_file'])[1])
                    image_path = os.path.join(src_dir, info_dict['raw_file'])
                    assert os.path.exists(
                        image_path), '{:s} not exist'.format(image_path)

                    h_samples = info_dict['h_samples']
                    lanes = info_dict['lanes']

                    for lane in lanes:
                        assert len(h_samples) == len(lane)
                        lane_pts = []
                        count = 0
                        for index in range(len(lane)):
                            ptx = lane[index]
                            pty = h_samples[index]
                            if ptx == -2:
                                ptz = 0
                            else:
                                ptz = 1
                                count += 1
                            lane_pts.append([ptx, pty, ptz])
                        # The label of lane pts has two types, len=48 and len=56
                        if len(lane_pts) == 48:
                            for k in range(8):
                                lane_pts.append([0, 0, 0])
                        if count > 15:
                            label_gt_pts.append(lane_pts)
                            label_image_path.append(image_path)

        return np.array(label_image_path), np.array(label_gt_pts)

    def __len__(self):
        return len(self._label_image_path)

    def __getitem__(self, idx):

        assert os.path.exists(self._label_image_path[idx]), '{:s} not exist'.format(
            self._label_image_path[idx])

        gt_image = cv2.imread(
            self._label_image_path[idx], cv2.IMREAD_UNCHANGED)
        self.original_shape = gt_image.shape
        gt_image = cv2.resize(gt_image, dsize=self.resize,
                              interpolation=cv2.INTER_LINEAR)
        gt_pts_lane = self._label_gt_pts[idx]

        # I do this for hnet model to work. todo: check if this is the right way
        gt_image = np.transpose(gt_image, (2, 0, 1))
        # resize gt_pts_lane
        width_ratio = self.resize[0] / self.original_shape[1]
        height_ratio = self.resize[1] / self.original_shape[0]
        gt_pts_lane = np.array(self._label_gt_pts[idx])
        gt_pts_lane[:, 0] = gt_pts_lane[:, 0] * width_ratio
        gt_pts_lane[:, 1] = gt_pts_lane[:, 1] * height_ratio

        return gt_image, gt_pts_lane


if __name__ == '__main__':
    # print some image and lane point. choose random image
    import matplotlib.pyplot as plt
    import random
    import time
    # train_data_set_path = '/home/talg/Downloads/train_set'
    train_data_set_path = '/home/tomer/Downloads/EE_master/Deep_Learning/Lane_Detection_Using_Perspective_Transformation_project/TUSimple/train_set'
    train_data_set = TusimpleForHnetDataSet(train_data_set_path)
    print("len(train_data_set): ", len(train_data_set))

    gt_image, lane_points = train_data_set[random.randint(
        0, len(train_data_set))]
    # draw lane points on image
    for lane in lane_points:
        if lane[2] == 1:
            src_image = gt_image.transpose(1, 2, 0)
            src_image = cv2.circle(
                src_image, (lane[0], lane[1]), 1, (0, 0, 255), -1)
    plt.imshow(src_image)
    plt.show()

    # todo: currenty wer'e getting 1 lane per image, (idea from reference code lanenet-hnet).
    # do we really want it? or do we want all lanes per image?
