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
import helper

class cnn_model:
    def setup_input(self):
        # placeholder for input and label
        x = tf.placeholder(tf.float32, [opt.train_batch_size, opt.input_h, opt.input_w, 3])
        gt = tf.placeholder(tf.float32, [opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])
        
        return x, gt

    # model definition
    def first_stage(self, image):    
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

    def stage_block(self, image, heatmap_approx):
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
        '''with tf.variable_scope("stage4"):
            stage_4_output = self.stage_block(image, stage_3_output)
        with tf.variable_scope("stage5"):
            stage_5_output = self.stage_block(image, stage_4_output)
        with tf.variable_scope("stage6"):
            stage_6_output = self.stage_block(image, stage_5_output)'''

        with tf.variable_scope("stage1"):
            stage_1_output_deconved = self.deconv(stage_1_output, opt.train_batch_size)
        with tf.variable_scope("stage2"):
            stage_2_output_deconved = self.deconv(stage_2_output, opt.train_batch_size)
        with tf.variable_scope("stage3"):
            stage_3_output_deconved = self.deconv(stage_3_output, opt.train_batch_size)
        '''with tf.variable_scope("stage4"):
            stage_4_output_deconved = self.deconv(stage_4_output, opt.train_batch_size)
        with tf.variable_scope("stage5"):
            stage_5_output_deconved = self.deconv(stage_5_output, opt.train_batch_size)
        with tf.variable_scope("stage6"):
            stage_6_output_deconved = self.deconv(stage_6_output, opt.train_batch_size)'''

        #loss = tf.reduce_mean(tf.nn.l2_loss(stage_1_output_deconved - gt)+tf.nn.l2_loss(stage_2_output_deconved - gt)+tf.nn.l2_loss(stage_3_output_deconved - gt)+tf.nn.l2_loss(stage_4_output_deconved - gt)+tf.nn.l2_loss(stage_5_output_deconved - gt)+tf.nn.l2_loss(stage_6_output_deconved - gt))
        loss = tf.reduce_mean(tf.nn.l2_loss(stage_1_output_deconved - gt)+tf.nn.l2_loss(stage_2_output_deconved - gt)+tf.nn.l2_loss(stage_3_output_deconved - gt))
        #log_variable(loss)
        train_step = tf.train.AdamOptimizer(opt.lr).minimize(loss)
        
        return stage_3_output_deconved, loss, train_step