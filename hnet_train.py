# reference: https://github.com/stesha2016/lanenet-enet-hnet/blob/master/lanenet_hnet_predict.py
import os
import os.path as ops
import numpy as np
import torch
import cv2
import time

from dataset.dataset_utils import TUSIMPLE, TUSIMPLE_AUG
from Hnet.hnet_model import HNet
from Hnet.hnet_loss import HNetLoss
from torch.autograd import Variable

HNET_TRAIN_OUTPUT_DIR = "TUSIMPLE/Hnet_output"
# Directory of train, valid, test .txt files
root = "TUSIMPLE/txt_for_local"

# Build train, valid, test set
train_set = TUSIMPLE(root=root, flag="train", resize=(128, 64))
valid_set = TUSIMPLE(root=root, flag="valid", resize=(128, 64))
test_set = TUSIMPLE(root=root, flag="test", resize=(128, 64))

# In case you want to run using the augmented dataset
# uncomment the following lines
# train_set = TUSIMPLE_AUG(root=root, flag='train')
# valid_set = TUSIMPLE_AUG(root=root, flag='valid')
# test_set = TUSIMPLE_AUG(root=root, flag='test')

# Print the number of samples of each set
print("train_set length {}".format(len(train_set)))
print("valid_set length {}".format(len(valid_set)))
print("test_set length {}".format(len(test_set)))

# Choice a random index to get the corresponding
# sample image, binary lane image, instance lane image
idx = 0
gt, bgt, igt = train_set[idx]

# Print some information about those images
print("image type {}".format(type(gt)))
print("image size {} \n".format(gt.size()))

print("gt binary image type {}".format(type(bgt)))
print("gt binary image size {}".format(bgt.size()))
print("items in gt binary image {} \n".format(torch.unique(bgt)))

print("gt instance type {}".format(type(igt)))
print("gt instance size {}".format(igt.size()))
print("items in gt instance {} \n".format(torch.unique(igt)))

# Define the batch size
batch_size = 10

# Define DataLoaders
data_loader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0
)
data_loader_valid = torch.utils.data.DataLoader(
    valid_set, batch_size=1, shuffle=True, num_workers=0
)
data_loader_test = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=0
)

# Define the learning rate
learning_rate = 5e-5

# Use GPU if available, else use CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define the model
hnet_model = HNet()
hnet_model.to(device)

# Define the optimizer
params = [p for p in hnet_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0002)

# Define scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1)

# Define number of epochs
num_epochs = 30

# Define the loss function
# criterion = HNetLoss()

# Define a list to append the losses
loss_all = []

# Let's train the model
for epoch in range(num_epochs):
    hnet_model.train()
    ts = time.time()
    for iter, batch in enumerate(data_loader_train):
        input_image = Variable(batch[0]).to(device)
        binary_labels = Variable(batch[1]).to(device)

        # Forward pass
        h_prediction = hnet_model(input_image)

        # Compute loss
        # get the lanes pixels by taking the "ones" pixels from binary_labels
        batch_size = binary_labels.shape[0]

        if torch.cuda.is_available():
            h_loss = torch.tensor(0.).cuda()
        else:
            h_loss = torch.tensor(0.)

        for i in range(batch_size):
            gt_lanes_pixels = torch.nonzero(binary_labels[i, :, :])
            # get the h_prediction (6 coeff) into a valid 3X3 homograpy matrix
            H = torch.zeros(batch_size, 3, 3)

            # create homogeneous coordinates from gt_lanes_pixels
            if torch.cuda.is_available():
                gt_lanes_pixels = torch.cat(
                    (gt_lanes_pixels, torch.ones(gt_lanes_pixels.shape[0], 1).cuda()), dim=1)
                H = torch.zeros(batch_size, 3, 3).cuda()
            else:
                gt_lanes_pixels = torch.cat(
                    (gt_lanes_pixels, torch.ones(gt_lanes_pixels.shape[0], 1)), dim=1)
                H = torch.zeros(batch_size, 3, 3)
            # assign the h_prediction to the H matrix
            H[:, 0, 0] = h_prediction[:, 0]  # a
            H[:, 0, 1] = h_prediction[:, 1]  # b
            H[:, 0, 2] = h_prediction[:, 2]  # c
            H[:, 1, 1] = h_prediction[:, 3]  # d
            H[:, 1, 2] = h_prediction[:, 4]  # e
            H[:, 2, 1] = h_prediction[:, 5]  # f
            H[:, -1, -1] = 1
            hnet_loss = HNetLoss(
                gt_lanes_pixels, H[i], "hnet_loss", usegpu=torch.cuda.is_available())
            h_loss += hnet_loss._hnet_loss()

        h_loss = h_loss / batch_size
        loss = h_loss
        optimizer.zero_grad()
        loss_all.append(loss.item())
        loss.backward()
        optimizer.step()
        if iter % 20 == 0:
            print("Epoch {} Iteration {} HLoss: {} ".format(epoch, iter, h_loss.item()))
    lr_scheduler.step()
    print("Epoch {} finished! Time elapsed {} ".format(epoch, time.time() - ts))
    os.makedirs(HNET_TRAIN_OUTPUT_DIR, exist_ok=True)
    torch.save(hnet_model.state_dict(), f"{HNET_TRAIN_OUTPUT_DIR}/hnet_epoch_{epoch}_batch_size_{batch_size}.model")

# Plot the loss
import matplotlib.pylab as plt
plt.plot(loss_all)
# save plot to file
plt.savefig(f"{HNET_TRAIN_OUTPUT_DIR}/hnet_loss_hnet_epoch_{epoch}_batch_size_{batch_size}.png")