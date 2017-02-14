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
# local import files
import opts as opt

# helper functions
def read_txt(file_name):
    inputLines = []
    with open(file_name) as inputStream:
        for separateLine in iter(inputStream):
            inputLines.append(separateLine.strip('\n'))
    return inputLines

class read_data:
    def __init__(self):
        # the list has already been shuffled
        #print('Reading Training File List ...')
        #self.imgList = read_txt(opt.img_list_txt)
        #print('Finished !')
        #self.nowIdx = 0
        self.cap = cv2.VideoCapture('/home/shangxuan/tf_pedestrian/video.mp4')

    def reset(self):
        # reset the index to 0
        self.nowIdx = 46

    def preprocess_img(self, img):
        # to-do: add random crop

        # resize
        img = cv2.resize(img, (opt.input_w, opt.input_h))
        # zero mean
        img = (img.astype(float) - 128)/128
        return img
    
    def get_next_train_batch(self):

        # batch placeholder
        batch_img = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, 3])
        batch_gt = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])

        for i in range(opt.train_batch_size):
            # get image
            #img_fn = self.imgList[self.nowIdx]
            #self.nowIdx = self.nowIdx + 1 if self.nowIdx < len(self.imgList) - 1 else 0
            # preprocess image
            #img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
            ret, frame = self.cap.read()
            # get the center part of frame
            frame = frame[316:765,661:1260,:]
            #
            frame = self.preprocess_img(frame)
            # get gt
            #gt_fn = img_fn.replace('image', 'heatmap_new_shangxuan')[:-4]+'.mat'
            #gt = sio.loadmat(gt_fn)['heatmap'] # 288*384*13
            batch_img[i,:,:,:] = frame
            #batch_gt[i,:,:,:] = gt
        
        # batch_img [-1,1], batch_gt [0,1]
        
        return batch_img, batch_gt
    def end(self):
        self.cap.release()

    def get_next_test_example(self):
    
        img_fn = self.imgList[self.nowIdx]
        self.nowIdx = self.nowIdx + 1 if self.nowIdx < len(self.imgList) -1 else 0
        img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        img = self.preprocess_img(img)
        # get gt
        gt_fn = img_fn.replace('image', 'heatmap_new_shangxuan')[:-4]+'.mat'
        gt = sio.loadmat(gt_fn)['heatmap'] # 288*284*13
        # reshape    
        img = np.reshape(img, [1, opt.input_h, opt.input_w, 3])
        gt = np.reshape(gt, [1, opt.input_h, opt.input_w, opt.nOutputs])
        
        return img, gt

class save_result:
    def __init__(self):
        # for storing result image
        self.dir = '/home/shangxuan/tf_pedestrian/real_results/'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        else:
            shutil.rmtree(self.dir)
            os.makedirs(self.dir)
        # for storing model
        #if not os.path.exists(opt.model_save_path):
        #    os.makedirs(opt.model_save_path)
        #else:
        #    shutil.rmtree(opt.model_save_path)
        #    os.makedirs(opt.model_save_path)

        self.nowIdx = 0
    
    def reset(self):
        self.nowIdx = 0

    def save_forward_result(self, original_image_batch, forward_results):
        original_image_batch = original_image_batch * 128 + 128
        forward_results = forward_results * 255
        for i in range(opt.train_batch_size):
            cv2.imwrite(self.dir + str(self.nowIdx+1) + '.png', np.reshape(original_image_batch[i,:,:,:], [opt.input_h, opt.input_w, 3]))
            for j in range(opt.nOutputs):
                cv2.imwrite(self.dir + str(self.nowIdx+1) + '_' + opt.joint_names[j] +'.png', np.reshape(forward_results[i,:,:,j], [opt.input_h, opt.input_w, 1]))
            self.nowIdx = self.nowIdx + 1

def save_forward_result(data):
    pass

def setup_input():
    # placeholder for input and label
    x = tf.placeholder(tf.float32, [opt.train_batch_size, opt.input_h, opt.input_w, 3])
    gt = tf.placeholder(tf.float32, [opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])
    
    return x, gt

# model definition
def first_stage(image):    
    # Create the model
    W_2 = tf.get_variable("W_2", shape=[9, 9, 3, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv1_2 = tf.nn.conv2d(image, W_2, strides=[1, 1, 1, 1], padding='SAME')
    relu1_2 = tf.nn.relu(conv1_2)
    mp1_2 = tf.nn.max_pool(relu1_2,[1,3,3,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

    W_3 = tf.get_variable("W_3", shape=[9, 9, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv1_3 = tf.nn.conv2d(mp1_2, W_3, strides=[1, 1, 1, 1], padding='SAME')
    relu1_3 = tf.nn.relu(conv1_3)
    mp1_3 = tf.nn.max_pool(relu1_3,[1,3,3,1],[1,2,2,1],padding='SAME')
    
    W_4 = tf.get_variable("W_4", shape=[9, 9, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv1_4 = tf.nn.conv2d(mp1_3, W_4, strides=[1, 1, 1, 1], padding='SAME')
    relu1_4 = tf.nn.relu(conv1_4)
    mp1_4 = tf.nn.max_pool(relu1_4,[1,3,3,1],[1,2,2,1],padding='SAME')
    
    W_5 = tf.get_variable("W_5", shape=[5, 5, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv1_5 = tf.nn.conv2d(mp1_4, W_5, strides=[1, 1, 1, 1], padding='SAME')
    relu1_5 = tf.nn.relu(conv1_5)

    W_6 = tf.get_variable("W_6", shape=[5, 5, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv1_6 = tf.nn.conv2d(relu1_5, W_6, strides=[1, 1, 1, 1], padding='SAME')
    relu1_6 = tf.nn.relu(conv1_6)

    W_7 = tf.get_variable("W_7", shape=[1, 1, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv1_7 = tf.nn.conv2d(relu1_6, W_7, strides=[1, 1, 1, 1], padding='SAME')
    relu1_7 = tf.nn.relu(conv1_7)

    W_8 = tf.get_variable("W_8", shape=[1, 1, 256, opt.nOutputs], initializer=tf.contrib.layers.xavier_initializer())
    conv1_8 = tf.nn.conv2d(relu1_7, W_8, strides=[1, 1, 1, 1], padding='SAME')
    relu1_8 = tf.nn.relu(conv1_8)

    return relu1_8

def stage_block(image, heatmap_approx):
    # first half
    Wx_2 = tf.get_variable("Wx_2", shape=[9,9,3,128], initializer=tf.contrib.layers.xavier_initializer())
    convx_2 = tf.nn.conv2d(image, Wx_2, strides=[1, 1, 1, 1], padding='SAME')
    relux_2 = tf.nn.relu(convx_2)
    mpx_2 = tf.nn.max_pool(relux_2,[1,3,3,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

    Wx_3 = tf.get_variable("Wx_3", shape=[9,9,128,128], initializer=tf.contrib.layers.xavier_initializer())
    convx_3 = tf.nn.conv2d(mpx_2, Wx_3, strides=[1, 1, 1, 1], padding='SAME')
    relux_3 = tf.nn.relu(convx_3)
    mpx_3 = tf.nn.max_pool(relux_3,[1,3,3,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

    Wx_4 = tf.get_variable("Wx_4", shape=[9,9,128,128], initializer=tf.contrib.layers.xavier_initializer())
    convx_4 = tf.nn.conv2d(mpx_3, Wx_4, strides=[1, 1, 1, 1], padding='SAME')
    relux_4 = tf.nn.relu(convx_4)
    mpx_4 = tf.nn.max_pool(relux_4,[1,3,3,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

    Wx_5 = tf.get_variable("Wx_5", shape=[5,5,128,opt.nOutputs], initializer=tf.contrib.layers.xavier_initializer())
    convx_5 = tf.nn.conv2d(mpx_4, Wx_5, strides=[1, 1, 1, 1], padding='SAME')
    relux_5 = tf.nn.relu(convx_5)

    # concatenate
    inter = tf.concat(3, [relux_5, heatmap_approx])
    
    # second half
    Wi_1 = tf.get_variable("Wi_1", shape=[11, 11, 2*opt.nOutputs, 128], initializer=tf.contrib.layers.xavier_initializer())
    convi_1 = tf.nn.conv2d(inter, Wi_1, strides=[1, 1, 1, 1], padding='SAME')
    relui_1 = tf.nn.relu(convi_1)

    Wi_2 = tf.get_variable("Wi_2", shape=[11, 11, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    convi_2 = tf.nn.conv2d(relui_1, Wi_2, strides=[1, 1, 1, 1], padding='SAME')
    relui_2 = tf.nn.relu(convi_2)

    Wi_3 = tf.get_variable("Wi_3", shape=[11, 11, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    convi_3 = tf.nn.conv2d(relui_2, Wi_3, strides=[1, 1, 1, 1], padding='SAME')
    relui_3 = tf.nn.relu(convi_3)

    Wi_4 = tf.get_variable("Wi_4", shape=[11, 11, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    convi_4 = tf.nn.conv2d(relui_3, Wi_4, strides=[1, 1, 1, 1], padding='SAME')
    relui_4 = tf.nn.relu(convi_4)

    Wi_5 = tf.get_variable("Wi_5", shape=[11, 11, 256, opt.nOutputs], initializer=tf.contrib.layers.xavier_initializer())
    convi_5 = tf.nn.conv2d(relui_4, Wi_5, strides=[1, 1, 1, 1], padding='SAME')
    relui_5 = tf.nn.relu(convi_5)

    return relui_5


def deconv(x, batch_size):
    output_shape = [batch_size,opt.input_h,opt.input_w,opt.nOutputs]
    W_deconv = tf.get_variable("W_deconv", shape=[17,17,opt.nOutputs,opt.nOutputs], initializer=tf.contrib.layers.xavier_initializer())
    deconved_x = tf.sigmoid(tf.nn.conv2d_transpose(x, W_deconv,tf.pack(output_shape),[1,8,8,1],padding='SAME'))
    return deconved_x

def final_stage():
    # to be added
    return

def setup_model(image, gt):
    with tf.variable_scope("stage1"):
        stage_1_output = first_stage(image)
    with tf.variable_scope("stage2"):
        stage_2_output = stage_block(image, stage_1_output)
    with tf.variable_scope("stage3"):
        stage_3_output = stage_block(image, stage_2_output)

    with tf.variable_scope("stage1"):
        stage_1_output_deconved = deconv(stage_1_output, opt.train_batch_size)
    with tf.variable_scope("stage2"):
        stage_2_output_deconved = deconv(stage_2_output, opt.train_batch_size)
    with tf.variable_scope("stage3"):
        stage_3_output_deconved = deconv(stage_3_output, opt.train_batch_size)

    loss = tf.reduce_mean(tf.nn.l2_loss(stage_1_output_deconved - gt)+tf.nn.l2_loss(stage_2_output_deconved - gt)+tf.nn.l2_loss(stage_3_output_deconved - gt))
    train_step = tf.train.AdamOptimizer(opt.lr).minimize(loss)
    
    return stage_3_output_deconved, loss, train_step

# main function
if __name__ == '__main__':
    # set up input and output
    x, gt = setup_input()
    # build model
    stage_3_output_deconved, loss, train_step = setup_model(x, gt)
    # new data fetch and save object
    data_fetch_obj = read_data()
    data_save_obj = save_result()
    # initialize variables
    sess = tf.InteractiveSession()
    #sess.run(tf.global_variables_initializer())
    # new tensorflow model saver object
    saver = tf.train.Saver()
    # start training and testing
    #for i in range(opt.epoch):
    #    # training
    #    data_fetch_obj.reset()
    #    for j in range(int(math.floor(opt.total_image_num / opt.train_batch_size))):
    #        batch_x, batch_y = data_fetch_obj.get_next_train_batch()
    #        print('Epoch %d Batch %d: loss %f' % (i+1, j+1, loss.eval(feed_dict={x: batch_x, gt: batch_y})))
    #        train_step.run(feed_dict={x: batch_x, gt: batch_y})
    #        # you can check particular tensor in pdb: "print(tf.get_default_graph().get_tensor_by_name("stage1/W_2:0").eval())", if checking the feature map, should feed_dict in the eval() 
    #    # testing (forwarding the first 10 training images to visualize the result)
    #    data_fetch_obj.reset()
    #    data_save_obj.reset()
    #    for j in range(opt.test_image_batch):
    #        print('Testing Batch %d' % j+1)
    #        batch_x, batch_y = data_fetch_obj.get_next_train_batch()
    #        batch_image_forward_result = stage_3_output_deconved.eval(feed_dict={x: batch_x, gt: batch_y})
    #        data_save_obj.save_forward_result(batch_x, batch_image_forward_result)
    #    # save the model for this epoch
    #    save_path = saver.save(sess, opt.model_save_path + 'model.ckpt')
    #print('Optimize Finished.')
    
    # restore variables
    saver.restore(sess, "/home/shangxuan/backup_tf_networks/tf_pedestrian_backup_4Jan/saved_model/model.ckpt")
    print("Model restored")

    for i in range(24):
        #print('Testing Batch %d' % i+1)
        batch_x, batch_y = data_fetch_obj.get_next_train_batch()
        batch_image_forward_result = stage_3_output_deconved.eval(feed_dict={x: batch_x, gt: batch_y})
        data_save_obj.save_forward_result(batch_x, batch_image_forward_result)
    
    data_fetch_obj.end()
    print('All saved in folder.')
   
    