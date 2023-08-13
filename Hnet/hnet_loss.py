# reference: https://github.com/klintan/pytorch-lanenet/blob/master/lanenet/model/loss.py

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
from torch.functional import F

class HNetLoss(_Loss):
    """
    HNet Loss
    """

    def __init__(self, gt_pts, transformation_coefficient, name, usegpu=True):
        """

        :param gt_pts: [x, y, 1]
        :param transformation_coeffcient: [[a, b, c], [0, d, e], [0, f, 1]]
        :param name:
        :return: 
        """
        super(HNetLoss, self).__init__()

        self.gt_pts = gt_pts

        self.transformation_coefficient = transformation_coefficient
        self.name = name
        self.usegpu = usegpu

    def _hnet_loss(self):
        """

        :return:
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)
        loss = torch.mean(torch.pow(self.gt_pts.t()[0, :] - x_transformation_back[0, :], 2))

        return loss

    def _hnet(self):
        """

        :return:
        """
        self.transformation_coefficient = torch.cat((self.transformation_coefficient, torch.tensor([1.0])),
                                                    dim=0)
        H_indices = torch.tensor([0, 1, 2, 4, 5, 7, 8])
        H_shape = 9
        H = torch.zeros(H_shape)
        H.scatter_(dim=0, index=H_indices, src=self.transformation_coefficient)
        H = H.view((3, 3))

        pts_projects = torch.matmul(H, self.gt_pts.t())

        Y = pts_projects[1, :]
        X = pts_projects[0, :]
        Y_One = torch.ones(Y.size())
        Y_stack = torch.stack((torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One), dim=1).squeeze()
        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(Y_stack.t(), Y_stack)),
                                      Y_stack.t()),
                         X.view(-1, 1))

        x_preds = torch.matmul(Y_stack, w)
        preds = torch.stack((x_preds.squeeze(), Y, Y_One), dim=1).t()
        return (H, preds)

    def _hnet_transformation(self):
        """
        """
        H, preds = self._hnet()
        x_transformation_back = torch.matmul(torch.inverse(H), preds)

        return x_transformation_back

    def forward(self, input, target, n_clusters):
        return self._hnet_loss(input, target)