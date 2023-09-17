# reference: https://github.com/klintan/pytorch-lanenet/blob/master/lanenet/model/loss.py

import torch
from torch.functional import F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from hnet_utils import hnet_transformation


class HNetLoss(_Loss):
    """
    HNet Loss
    """
    def __init__(self):
        super(HNetLoss, self).__init__()

    def forward(self, input_pts, transformation_coefficient):
        return self._hnet_loss(input_pts, transformation_coefficient)

    def _hnet_loss(self, input_pts, transformation_coefficient):
        """
        :return:
        """
        # H, preds = self._hnet()
        # x_transformation_back = torch.matmul(torch.inverse(H), preds)
        # loss = torch.mean(torch.pow(self.gt_pts.t()[0, :] - x_transformation_back[0, :], 2))

        x_preds_transformation_back = hnet_transformation(input_pts, transformation_coefficient, device='cuda')

        # compute loss between back-transformed polynomial fit and gt_pts
        loss = torch.mean(torch.pow(input_pts.t()[0, :] - x_preds_transformation_back[0, :], 2))

        return loss
