import torch
import numpy as np
from torch.functional import F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

# Use GPU if available, else use CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

PRE_H = np.array([-2.04835137e-01, -3.09995252e+00, 7.99098762e+01, -
                 2.94687413e+00, 7.06836681e+01, -4.67392998e-02]).astype(np.float32)
PRE_H = torch.from_numpy(PRE_H).to(device)


class PreTrainHnetLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def pre_train_loss(self, transformation_coefficient):
        """
        :param transformation_coefficient: the 6 params from the HNet
        :return: the loss
        """
        # todo, I add the dim=1 to get the mean of the batch, but I'm not sure if it's correct
        # if it wasn't there, I would get scalar after the norm and the mean would be the same
        pre_loss = torch.mean(torch.norm((transformation_coefficient - PRE_H) / PRE_H, dim=1))
        return pre_loss

    def forward(self, transformation_coefficient):
        return self.pre_train_loss(transformation_coefficient)
    

def hnet_transformation(gt_pts, transformation_coeffcient, order=3):
    """

    :param gt_pts:
    :param transformation_coeffcient:
    :return:
    """
    transformation_coeffcient = torch.concat(torch.squeeze(transformation_coeffcient), [1.0], axis=-1)
    # multiplier = torch.constant([1., 1., 4., 1., 4., 0.25, 1.])
    # transformation_coeffcient = transformation_coeffcient * multiplier
    # H_indices = torch.tensor([[0], [1], [2], [4], [5], [7], [8]])
    # H_shape = torch.tensor([9])
    # H = torch.scatter(H_indices, transformation_coeffcient, H_shape)
    # H = torch.reshape(H, shape=[3, 3])

    H = torch.zeros(1, 3, 3)
    # assign the h_prediction to the H matrix
    H[:, 0, 0] = transformation_coeffcient[:, 0]  # a
    H[:, 0, 1] = transformation_coeffcient[:, 1]  # b
    H[:, 0, 2] = transformation_coeffcient[:, 2]  # c
    H[:, 1, 1] = transformation_coeffcient[:, 3]  # d
    H[:, 1, 2] = transformation_coeffcient[:, 4]  # e
    H[:, 2, 1] = transformation_coeffcient[:, 5]  # f
    H[:, -1, -1] = 1


    pts_projects = torch.matmul(H, gt_pts)

    Y = torch.transpose(pts_projects[1, :] / pts_projects[2, :])
    X = torch.transpose(pts_projects[0, :] / pts_projects[2, :])
    Y_One = torch.ones_like(Y)
    if order == 2:
        Y_stack = torch.stack([torch.pow(Y, 2), Y, Y_One], axis=1)
    elif order == 3:
        Y_stack = torch.stack([torch.pow(Y, 3), torch.pow(Y, 2), Y, Y_One], axis=1)
    else:
        raise ValueError('Unknown order', order)

    
    w = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(Y_stack.transpose(0, 1), Y_stack)),
                                      Y_stack.transpose(0, 1)), X.unsqueeze(1))

    x_preds = torch.matmul(Y_stack, w)
    preds = torch.transpose(torch.stack([torch.squeeze(x_preds, -1) * pts_projects[2, :],
                                    Y * pts_projects[2, :], pts_projects[2, :]], axis=1))
    x_transformation_back = torch.matmul(torch.matrix_inverse(H), preds)

    # extra returns for use to use
    pts_projects_nomalized = pts_projects / pts_projects[2, :]

    return x_transformation_back, H, pts_projects_nomalized
