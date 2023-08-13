# reference: https://github.com/stesha2016/lanenet-enet-hnet/blob/master/lanenet_hnet_predict.py
import os.path as ops
import numpy as np
import torch
import cv2
import time

from dataset.dataset_utils import TUSIMPLE, TUSIMPLE_AUG
from Hnet.hnet_model import HNet
from Hnet.hnet_loss import HNetLoss
from torch.autograd import Variable

# Directory of train, valid, test .txt files 
root = 'TUSIMPLE/txt_for_local'

# Build train, valid, test set
train_set = TUSIMPLE(root=root, flag='train', resize=(128,72))
valid_set = TUSIMPLE(root=root, flag='valid', resize=(128,72))
test_set = TUSIMPLE(root=root, flag='test', resize=(128,72))

# In case you want to run using the augmented dataset
# uncomment the following lines
# train_set = TUSIMPLE_AUG(root=root, flag='train')
# valid_set = TUSIMPLE_AUG(root=root, flag='valid')
# test_set = TUSIMPLE_AUG(root=root, flag='test')

# Print the number of samples of each set
print('train_set length {}'.format(len(train_set)))
print('valid_set length {}'.format(len(valid_set)))
print('test_set length {}'.format(len(test_set)))

# Choice a random index to get the corresponding
# sample image, binary lane image, instance lane image
idx = 0 
gt, bgt, igt = train_set[idx]

# Print some information about those images
print('image type {}'.format(type(gt)))
print('image size {} \n'.format(gt.size()))

print('gt binary image type {}'.format(type(bgt)))
print('gt binary image size {}'.format(bgt.size()))
print('items in gt binary image {} \n'.format(torch.unique(bgt)))

print('gt instance type {}'.format(type(igt)))
print('gt instance size {}'.format(igt.size()))
print('items in gt instance {} \n'.format(torch.unique(igt)))

# Define the batch size
batch_size = 8

# Define DataLoaders
data_loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
data_loader_valid = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=0)
data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

# Define the learning rate
learning_rate = 5e-4

# Use GPU if available, else use CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the model
hnet_model = HNet()
hnet_model.to(device)

# Define the optimizer
params = [p for p in hnet_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0002)

# Define scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
        # for gt we take the binary_labels with ones as GT marking lane point 
        # and get them to shape [batch_size, [x, y, 1]]
        gt_labels = binary_labels.view(batch_size, -1, 3)[:, :, 0:2]

        loss = HNetLoss(h_prediction, binary_labels)

        # loss = hnet_loss
        optimizer.zero_grad()
        loss_all.append(loss.item())
        loss.backward()
        optimizer.step()

    lr_scheduler.step()
