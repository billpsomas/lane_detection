import os.path as ops
import numpy as np
import torch
import cv2
import time

from dataset.dataset_utils import TUSIMPLE, TUSIMPLE_AUG
from Lanenet.model import Lanenet
from Lanenet.cluster_loss import cluster_loss
from torch.autograd import Variable

# Directory of train, valid, test .txt files 
root = 'TUSIMPLE/txt_for_local'

# Build train, valid, test set
train_set = TUSIMPLE(root=root, flag='train')
valid_set = TUSIMPLE(root=root, flag='valid')
test_set = TUSIMPLE(root=root, flag='test')

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
LaneNet_model = Lanenet(2, 4)
LaneNet_model.to(device)

# Define the optimizer
params = [p for p in LaneNet_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0002)

# Define scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define number of epochs
num_epochs = 30

# Define the loss function
criterion = cluster_loss()

# Define a list to append the losses 
loss_all = []

# Let's train the model
for epoch in range(num_epochs):
    LaneNet_model.train()
    ts = time.time()
    for iter, batch in enumerate(data_loader_train):
        input_image = Variable(batch[0]).to(device)
        binary_labels = Variable(batch[1]).to(device)
        instance_labels = Variable(batch[2]).to(device)
        
        binary_final_logits, instance_embedding = LaneNet_model(input_image)
        # loss = LaneNet_model.compute_loss(binary_logits=binary_final_logits, binary_labels=binary_labels,
        #                               instance_logits=instance_embedding, instance_labels=instance_labels, delta_v=0.5, delta_d=3)
        binary_segmenatation_loss, instance_segmenatation_loss = criterion(binary_logits=binary_final_logits, binary_labels=binary_labels,
                                       instance_logits=instance_embedding, instance_labels=instance_labels, delta_v=0.5, delta_d=3)
        
        # binary_segmenatation_loss = criterion(binary_final_logits, binary_labels)
        loss = 1*binary_segmenatation_loss + 1*instance_segmenatation_loss
        optimizer.zero_grad()
        loss_all.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if iter % 20 == 0:
            print("Epoch {} Iteration {} Binary Segmentation Loss: {} Instance Segmentation Loss: {} ".format(epoch, iter, binary_segmenatation_loss.item(), instance_segmenatation_loss.item()))
    lr_scheduler.step()
    print("Epoch {} finished! Time elapsed {} ".format(epoch, time.time() - ts))
    torch.save(LaneNet_model.state_dict(), f"TUSIMPLE/Lanenet_output/lanenet_epoch_{epoch}_batch_size_{batch_size}.model")

# Plot the loss
import matplotlib.pylab as plt
plt.plot(loss_all)