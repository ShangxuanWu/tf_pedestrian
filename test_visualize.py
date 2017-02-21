# global import files
import argparse
import sys
import pdb
import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio
import scipy
import math
import shutil
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# local import files
import opt
import helper
import test_cityscape
import model_3stage

file_list_to_test = "./file_lists/train_list.txt"
 
class read_data:
    def __init__(self):
        # the list has already been shuffled
        print('Reading Training File List ...')
        self.trainImgList = helper.read_txt(opt.train_img_list_txt)
        self.testImgList = helper.read_txt(file_list_to_test)
        print('Finished !')
        self.nowTrainIdx = 0
        self.nowTestIdx = 0

    def reset(self):
        # reset the index to 0
        self.nowTestIdx = 0
        self.nowTrainIdx = 0
    
    def get_test_filelist_size(self):
        return len(self.testImgList)
    
    
    def get_next_train_batch(self):
        
        # batch placeholder
        batch_img = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, 3])
        batch_gt = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])

        for i in range(opt.train_batch_size):
            # get image
            img_fn = self.trainImgList[self.nowTrainIdx]
            self.nowTrainIdx = self.nowTrainIdx + 1
            # preprocess image
            img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
            img = helper.preprocess_img(img)
            # get gt
            gt_fn = img_fn.replace('image', 'gt')[:-4]+'.mat'
            gt = sio.loadmat(gt_fn)['heatmap'] # 288*284*channel
            batch_img[i,:,:,:] = img
            batch_gt[i,:,:,:] = gt
        
        # batch_img [-1,1], batch_gt [0,1]
        return batch_img, batch_gt

    def get_next_test_batch(self):
        
        # batch placeholder
        batch_img = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, 3])
        batch_gt = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])

        for i in range(opt.train_batch_size):
            # get image
            img_fn = self.testImgList[self.nowTestIdx]
            self.nowTestIdx = self.nowTestIdx + 1
            # preprocess image
            img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
            img = helper.preprocess_img(img)

            batch_img[i,:,:,:] = img
        
        # batch_img [-1,1], batch_gt [0,1]
        return batch_img, batch_gt

class save_result:
    def __init__(self):

        # for storing model
        if not os.path.exists(opt.model_save_path):
            os.makedirs(opt.model_save_path)
        else:
            shutil.rmtree(opt.model_save_path)
            os.makedirs(opt.model_save_path)
        
        # for storing log_variable
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        else:
            shutil.rmtree(opt.log_dir)
            os.makedirs(opt.log_dir)

        self.nowIdx = 0
    
    def reset(self, epoch):
        
        self.nowIdx = 0

        # for storing result image
        if not os.path.exists(opt.save_dir+str(epoch)):
            os.makedirs(opt.save_dir+str(epoch))
        else:
            shutil.rmtree(opt.save_dir+str(epoch))
            os.makedirs(opt.save_dir+str(epoch))

    def save_forward_result(self, original_image_batch, forward_results, epoch):
        # de-normalizing and saving
        original_image_batch = original_image_batch * 128 + 128
        # thresholding
        #pdb.set_trace()
        #forward_results = forward_results < opt.threshold        
        forward_results = forward_results * 255
        for i in range(opt.train_batch_size):
            # save original image
            cv2.imwrite(opt.save_dir + str(epoch) + '/' + str(self.nowIdx+1) + '.png', np.reshape(original_image_batch[i,:,:,:], [opt.input_h, opt.input_w, 3]))
            # save forward result
            for j in range(opt.nOutputs):
                cv2.imwrite(opt.save_dir + str(epoch) + '/' + str(self.nowIdx+1) + '_' + opt.joint_names[j] +'.png', np.reshape(forward_results[i,:,:,j], [opt.input_h, opt.input_w, 1]))
            self.nowIdx = self.nowIdx + 1



# this function is for logging
def log_variable(var):
    tf.summary.scalar('loss', var)

# main function
if __name__ == '__main__':
    
    model = model_3stage.cnn_model()

    # set up input and output
    x, gt = model.setup_input()
    
    # build model
    final_stage_output_deconved, loss, train_step = model.setup_model(x, gt)
    
    # new data fetch and save object
    data_fetch_obj = read_data()
    data_save_obj = save_result()
    #cityscape = test_cityscape.cityscape_data()
    
    # start session
    sess = tf.InteractiveSession()

    # initialize all
    sess.run(tf.global_variables_initializer())
    
    # new tensorflow model saver object
    saver = tf.train.Saver()
    
    # restore variables
    if 1:
        saver.restore(sess, '/home/shangxuan/tf_pedestrian/saved_model/model.ckpt')
        print("Model under this folder restored.")

    data_save_obj.reset(1)
    for j in range(50): # 10 images per batch
        print('Testing Batch %d' % (j+1))
        batch_x, batch_y = data_fetch_obj.get_next_test_batch()
        batch_image_forward_result = final_stage_output_deconved.eval(feed_dict={x: batch_x, gt: batch_y})

        # save forward results
        data_save_obj.save_forward_result(batch_x, batch_image_forward_result, 1)
