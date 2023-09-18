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


class TrainHetLoss(_Loss):

    def __init__(self):
        super(TrainHetLoss, self).__init__()

    def forward(self, input_pts, transformation_coefficient):
        return self._hnet_loss(input_pts, transformation_coefficient)

    @staticmethod
    def _hnet_loss(input_pts, transformation_coefficient):
        x_preds_transformation_back, _, _ = hnet_transformation(input_pts, transformation_coefficient, device='cuda')

        # compute loss between back-transformed polynomial fit and gt_pts
        loss = torch.mean(torch.pow(input_pts.t()[0, :] - x_preds_transformation_back[0, :], 2))

        return loss
