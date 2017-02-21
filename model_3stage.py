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
import opt
import helper

class cnn_model:
    def setup_input(self):
        # placeholder for input and label
        x = tf.placeholder(tf.float32, [opt.train_batch_size, opt.input_h, opt.input_w, 3])
        gt = tf.placeholder(tf.float32, [opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])
        
        return x, gt
    
    def read_original_model(self):
        self.params_original = np.load(opt.original_cpm_model, encoding='latin1').item()
        pass
    
    def Conv2D(self, name_, input_, shape_, stride_, with_CPM_initialize_=False, with_relu_=True):
        if with_CPM_initialize_:
            W = tf.get_variable(name, shape=shape_, initializer=tf.constant(self.params_original[name + '/W']))
            b = tf.get_variable(name, shape=[shape_[-1]], initializer=tf.constant(self.params_original[name + '/b']))
        else:
            W = tf.get_variable(name, shape=shape_, initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name, shape=[shape_[-1]], initializer=tf.contrib.layers.xavier_initializer())
        
        conv = tf.nn.conv2d(input_, W, strides=[stride_, stride_, stride_, stride_], padding='SAME')
        conv_plus_bias = tf.nn.bias_add(conv, b)
        if with_relu_:
            relu = tf.nn.relu(conv)
        return relu

    # model definition
    def first_stage(self, image):    
        # Create the model
        conv1_1 = self.Conv2D('conv1_1', image, [3, 3, 3, 64], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv1_2 = self.Conv2D('conv1_2', conv1_1, [3, 3, 64, 64], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        pool_1 = tf.nn.max_pool(conv1_2,[1,2,2,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

        conv2_1 = self.Conv2D('conv2_1', pool_1, [3, 3, 64, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv2_2 = self.Conv2D('conv2_2', conv2_1, [3, 3, 128, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        pool_2 = tf.nn.max_pool(conv1_2,[1,2,2,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

        conv3_1 = self.Conv2D('conv3_1', pool_2, [3, 3, 128, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv3_2 = self.Conv2D('conv3_2', conv3_1, [3, 3, 256, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv3_3 = self.Conv2D('conv3_3', conv3_2, [3, 3, 256, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv3_4 = self.Conv2D('conv3_4', conv3_3, [3, 3, 256, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        pool_3 = tf.nn.max_pool(conv3_4,[1,2,2,1],[1,2,2,1],padding='SAME') # ksize, stride, pool

        conv4_1 = self.Conv2D('conv4_1', pool_3, [3, 3, 256, 512], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv4_2 = self.Conv2D('conv4_2', conv4_1, [3, 3, 512, 512], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv4_3 = self.Conv2D('conv4_3_CPM', conv4_2, [3, 3, 512, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv4_4 = self.Conv2D('conv4_4_CPM', conv4_3, [3, 3, 256, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv4_5 = self.Conv2D('conv4_5_CPM', conv4_4, [3, 3, 256, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv4_6 = self.Conv2D('conv4_6_CPM', conv4_5, [3, 3, 256, 256], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv4_7 = self.Conv2D('conv4_7_CPM', conv4_6, [3, 3, 256, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)

        conv5_1 = self.Conv2D('conv5_1_CPM', conv4_7, [3, 3, 128, 512], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        conv5_2 = self.Conv2D('conv5_2_CPM', conv5_1, [3, 3, 512, 15], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        
        return relu1_8

    def stage_block(self, image, heatmap_approx, stage_name):
        Mconv1 = self.Conv2D('Mconv1'+'_'+stage_name, image, [7, 7, 144, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        Mconv2 = self.Conv2D('Mconv2'+'_'+stage_name, Mconv1, [7, 7, 128, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        Mconv3 = self.Conv2D('Mconv3'+'_'+stage_name, Mconv2, [7, 7, 128, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        Mconv4 = self.Conv2D('Mconv4'+'_'+stage_name, Mconv3, [7, 7, 128, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        Mconv5 = self.Conv2D('Mconv5'+'_'+stage_name, Mconv4, [7, 7, 128, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        Mconv6 = self.Conv2D('Mconv6'+'_'+stage_name, Mconv5, [7, 7, 128, 128], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=True)
        Mconv7 = self.Conv2D('Mconv7'+'_'+stage_name, Mconv6, [7, 7, 128, 15], 1, with_CPM_initialize_=opt.load_original_CPM, with_relu_=False)

        return Mconv7
        
    def deconv(self, x, batch_size):
        output_shape = [batch_size,opt.input_h,opt.input_w,opt.nOutputs]
        W_deconv = tf.get_variable("W_deconv", shape=[17,17,opt.nOutputs,opt.nOutputs], initializer=tf.contrib.layers.xavier_initializer())
        deconved_x = tf.sigmoid(tf.nn.conv2d_transpose(x, W_deconv,tf.pack(output_shape),[1,8,8,1],padding='SAME'))
        return deconved_x

    def setup_model(self, image, gt):
        with tf.variable_scope("stage1"):
            stage_1_output = self.first_stage(image)
        with tf.variable_scope("stage2"):
            stage_2_output = self.stage_block(image, stage_1_output)
        with tf.variable_scope("stage3"):
            stage_3_output = self.stage_block(image, stage_2_output)

        with tf.variable_scope("stage1"):
            stage_1_output_deconved = self.deconv(stage_1_output, opt.train_batch_size)
        with tf.variable_scope("stage2"):
            stage_2_output_deconved = self.deconv(stage_2_output, opt.train_batch_size)
        with tf.variable_scope("stage3"):
            stage_3_output_deconved = self.deconv(stage_3_output, opt.train_batch_size)

        loss = tf.reduce_mean(tf.nn.l2_loss(stage_1_output_deconved - gt)+tf.nn.l2_loss(stage_2_output_deconved - gt)+tf.nn.l2_loss(stage_3_output_deconved - gt))
        
        train_step = tf.train.AdamOptimizer(opt.lr).minimize(loss)
        
        return stage_3_output_deconved, loss, train_step