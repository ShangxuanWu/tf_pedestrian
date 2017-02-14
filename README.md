# How to run
`./run.sh`

# How to set default gpu?
modify run.sh

# Test policy?
- see [Here](https://www.cityscapes-dataset.com/), now using gtFine_trainvaltest validation set.
- user name socter@gmail.com
- pw 16720

# Useless files
- test_mp4.py (no longer using a test video)
- logs/* (no longer using tf.summary)
- test_image/* (no longer using several images to test)

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