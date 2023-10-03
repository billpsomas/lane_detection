import torch
import numpy as np
from torch.nn.modules.loss import _Loss

from hnet_utils import hnet_transformation


# Use GPU if available, else use CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

PRE_H = np.array([-2.04835137e-01, -3.09995252e+00, 7.99098762e+01, -
2.94687413e+00, 7.06836681e+01, -4.67392998e-02]).astype(np.float32)
PRE_H = torch.from_numpy(PRE_H).to(device)


class PreTrainHnetLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, transformation_coefficient):
        return self._pre_train_loss(transformation_coefficient)

    @staticmethod
    def _pre_train_loss(transformation_coefficient):
        """
        :param transformation_coefficient: the 6 params from the HNet
        :return: the loss
        """
        # todo, I add the dim=1 to get the mean of the batch, but I'm not sure if it's correct
        # if it wasn't there, I would get scalar after the norm and the mean would be the same
        pre_loss = torch.mean(torch.norm((transformation_coefficient - PRE_H) / PRE_H, dim=1))
        return pre_loss


class HetLoss(_Loss):

    def __init__(self):
        super(HetLoss, self).__init__()

    def forward(self, input_pts, transformation_coefficient, poly_fit_order: int = 3):
        return self._hnet_loss(input_pts, transformation_coefficient, poly_fit_order)

    def _hnet_loss(self, input_pts, transformation_coefficient, poly_fit_order: int = 3):
        # assert not torch.isnan(transformation_coefficient).any(), "transformation_coefficient is nan"
        # todo: handle case where transformation_coefficient is nan
        if torch.isnan(transformation_coefficient).any():
            print("transformation_coefficient is nan")
            return -1

        batch_size = input_pts.shape[0]
        single_frame_losses = []
        for i in range(batch_size):
            frame_input_pts = input_pts[i]
            frame_transformation_coefficient = transformation_coefficient[i]
            frame_loss = self.hnet_single_frame_loss(frame_input_pts, frame_transformation_coefficient, poly_fit_order)
            single_frame_losses.append(frame_loss)

        loss = torch.mean(torch.stack(single_frame_losses))

        return loss

    @staticmethod
    def hnet_single_frame_loss(input_pts, transformation_coefficient, poly_fit_order: int = 3):
        """
        :param input_pts: the points of the lane of a single image, shape: [k, 3] (k is the number of points)
        :param transformation_coefficient: the 6 params from the HNet, shape: [1, 6]
        :param poly_fit_order: the order of the polynomial
        :return single_frame_loss: the loss of the single frame
        """

        valid_pts_reshaped, _, preds_transformation_back, _ = hnet_transformation(input_pts,
                                                                                  transformation_coefficient,
                                                                                  poly_fit_order)
        # compute loss between back-transformed polynomial fit and gt_pts
        single_frame_loss = torch.mean(torch.pow(valid_pts_reshaped[0, :] - preds_transformation_back[0, :], 2))

        return single_frame_loss
