# TODO

- [ ] document new functions before merge to master
- [X] have main function with parameters (poly order, model path, etc) to the evaluate file
  - [X] have main with model path 
  - [X] have main with poly order
- [ ] add linesample to the transformed polyfit points to increase the number of points for the transformed back
  - [ ] use domain of the given points in lanenet-cluster. for the evaluation part
- [X] save when hnet is train the poly order and use it where ever needed (Hnet/train_hnet_v2.py:152)
- [ ] add init_seed function to train process (already exists in other branch - tomer's branch)
- [X] fix evaluation plotting function (set subplots size, add titles, fix image colors, add another image of polynomial samples)
- [X] hnet_transformation: consider using original sized points for and scale down the homography coeffs ( see reference lanenet-enet lanenet_model/hnet_loss.py line 80)
- [ ] check samples that creates very large hnet loss
  - [ ] draw gt images that have very large loss and see if they are messed up 
- [ ] during training, ignore batches with very large hnet loss (don't do optimizer.step)
- [ ] add regularization to hnet (tomer's branch)
  - [ ] we need to add prefix namings to files (weights, plots, images) according to the description:
    - [X] epoch (done)s
    - [X] poly order (done)
    - [ ] with/without regularization 
    - [ ] batch_size
- [X] when reading the weights we need to automatically use to correct poly order 
  - [X] it means we need to change the dict that is saved (no longer just the weights)
- [X] when fitting from lane clustering (in evaulutaion and predict) (Hnet/hnet_utils.py: 149) consider using everage of the lanenetcluster pixel  in each height (instead using them all)
- [X] consider connect through line all the points from poly in image plan after fit - decided not to do it, not going to help
- [ ] consider add RANSAC to points in fit during inference and also maybe in training. (in hnet_transformation?)
- [X] evaluate lanenet cluster and hnet at the same time with different preds fiels - to compare
  - [X] save visu debug images for both in evaluation 

# TODO if we have time
- [ ] train with all lanes in the same image at one 
  - [ ] requires different dataloader 

# innovation
 - [ ] pretrain
 - [ ] add regularization to hnet
 - [ ] tensorflow to pytorch
 - [ ] 