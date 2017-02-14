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
import opts as opt
import helper
import test_cityscape
import model_3stage
#import logger

# 
class read_data:
    def __init__(self):
        # the list has already been shuffled
        print('Reading Training File List ...')
        self.trainImgList = helper.read_txt(opt.train_img_list_txt)
        self.testImgList = helper.read_txt(opt.test_img_list_txt)
        print('Finished !')
        self.nowTrainIdx = 0
        self.nowTestIdx = 0

    def reset(self):
        # reset the index to 0
        self.nowTestIdx = 0
        self.nowTrainIdx = 0
    
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
        original_image_batch = original_image_batch * 128 + 128
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
    #data_save_obj = save_result()
    cityscape = test_cityscape.cityscape_data()
    
    # start session
    sess = tf.InteractiveSession()
    
    # start logger
    #merged = tf.summary.merge_all()
    #train_writer = tf.train.SummaryWriter(opt.log_dir + '/train', sess.graph)
    
    # another logger using matplotlib
    loss_record_stack = []

    # initialize all
    sess.run(tf.global_variables_initializer())
    
    # new tensorflow model saver object
    saver = tf.train.Saver()
    
    # restore variables
    if opt.load_pretrain:
        saver.restore(sess, opt.pretrain_dir)
        print("Model restored.")

    # start training and testing
    for i in range(opt.epoch):    
        # training
        data_fetch_obj.reset()
        for j in range(int(math.floor(opt.total_image_num / opt.train_batch_size))):
            batch_x, batch_y = data_fetch_obj.get_next_train_batch()
            this_batch_loss = loss.eval(feed_dict={x: batch_x, gt: batch_y})
            # logger
            loss_record_stack.append(this_batch_loss)
            # print loss
            print('Epoch %d Batch %d: loss %f' % (i+1, j+1, this_batch_loss))
            
            # get summary and write log
            #summary = merged.eval(feed_dict={x: batch_x, gt: batch_y})
            #train_writer.add_summary(summary, i+1)
            
            # back-propagation
            train_step.run(feed_dict={x: batch_x, gt: batch_y})
            
            # you can check particular tensor in pdb: "print(tf.get_default_graph().get_tensor_by_name("stage1/W_2:0").eval())", if checking the feature map, should feed_dict in the eval() 
        
        # testing (forwarding the selected 20 testing images to visualize the result)
        #data_save_obj.reset(i+1)
        cityscape.reset()
        for j in range(50): # 10 images per batch, 500 validation images in cityscape
            print('Testing Batch %d' % (j+1))
            #batch_x, batch_y = data_fetch_obj.get_next_test_batch()
            batch_x, batch_y = cityscape.get_next_test_batch()
            batch_image_forward_result = final_stage_output_deconved.eval(feed_dict={x: batch_x, gt: batch_y})

            # save forward results
            cityscape.save_forward_result(batch_image_forward_result, j)
            
        # save the model for this epoch
        save_path = saver.save(sess, opt.model_save_path)
        print('Epoch %d model saved !' % (i+1))

        # save plot for training loss
        loss_np_array = np.array(loss_record_stack)
        plt.plot(loss_np_array)
        plt.savefig(opt.loss_plot_path)
        print('Epoch %d loss plotted !' % (i+1))

        # test cityscape for formal accuracy
        cityscape.get_eval_set_result()

    print('Optimize Finished. Training Data Logged.')
