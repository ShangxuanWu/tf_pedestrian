# How to run
`./run.sh`

# How to set default gpu?
modify run.sh

# Test policy?
- see [Here](https://www.cityscapes-dataset.com/), now using gtFine_trainvaltest validation set.
- user name socter@gmail.com
- pw 16720

# Train and test files?
- Train files loaded in `main.py`, using 50000 synthetic images and 15 channel of ground truth (1 segmentation + 14 joints)
- Test files loaded in `test_cityscapes.py`, using 500 real images from CityScape validation set.

# What's inside logs?
- CityScape IoU testing result of every epoch

# Useless files
- test_mp4.py (no longer using a test video)
- test_image/* (no longer using several images to test)

# Feb 28th
## Add
1. Asynchroize data loading (reducing half of the training time, using TFRecord)
2. All-new CPM model and pretrain (Using some combination of old CPM and new CPM)

## new TFRecord training data link
scp -r /mnt/sdc1/shangxuan/synthetic3_tf/*

# Feb 21th
## Add 
1. load original CPM pretrain parameter
2. Debugged formal evaluation of CITYSCAPE
3. Changed the network structure from CPM to Multi-person CPM (see [Here](https://github.com/shihenw/convolutional-pose-machines-release/blob/master/model/_trained_MPI/pose_deploy_resize.prototxt))

# Feb 12th
## Add
1. Use new training dataset from Unity
2. Formal evaluation from CITYSCAPE
3. New file structures and class definitions
4. Save/load pre-trained models

# Jan 28th
## Add
1. global pooling layer -> from ParseNet

## Corrected
1. testing process, wrong output filename & results
2. change 'SET_VISIBLE_GPU' to explicit constraint
