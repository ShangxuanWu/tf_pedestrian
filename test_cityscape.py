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

# labels:
# 24: pedestrian
# 25: rider
# 33: bike

# 50 epoches to finish a test:


class cityscape_data:
    def __init__(self):
        
        # no shuffle
        print('Reading CityScape Validation File List ...')
        self.testImgList = helper.read_txt(opt.test_img_list_txt_cityscape)
        print('Finished !')
        self.nowTestIdx = 0

    def get_next_test_batch(self):

        # batch placeholder
        batch_img = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, 3])
        batch_gt = np.zeros([opt.train_batch_size, opt.input_h, opt.input_w, opt.nOutputs])

        # fill the blob with images
        for i in range(opt.train_batch_size/2):
            # get image
            img_fn = self.testImgList[self.nowTestIdx]
            self.nowTestIdx = self.nowTestIdx + 1
            # preprocess image
            img = cv2.imread(img_fn, cv2.IMREAD_COLOR)
            img = helper.preprocess_img_cut_half(img)

            batch_img[2*i:2*i+2,:,:,:] = img
        
        # batch_img [-1,1], batch_gt [0,1]
        return batch_img, batch_gt

    # run and save all the eval images
    def save_forward_result(self, forward_results, now_batch_minus_one, original_image_batch):
        forward_results = forward_results * 255
        for i in range(opt.train_batch_size/2):
            # save forward result, only segmentation, need upsample
            forward_file_name = self.testImgList[now_batch_minus_one * opt.train_batch_size/2 + i].replace('leftImg8bit','results')
            forward_file_dir = os.path.dirname(forward_file_name)
            os.system('mkdir -p '+ forward_file_dir)
            # clip left and right images into one image
            resized_forward_result_left = np.reshape(forward_results[2*i,:,:,0], [opt.input_h, opt.input_w])
            resized_forward_result_right = np.reshape(forward_results[2*i+1,:,:,0], [opt.input_h, opt.input_w])
            resized_forward_result = np.zeros([opt.input_h,opt.input_w*2,1])
            resized_forward_result[:,0:opt.input_w,0] = resized_forward_result_left
            resized_forward_result[:,opt.input_w:2*opt.input_w,0] = resized_forward_result_right
            resized_forward_result = cv2.resize(resized_forward_result, (opt.cityscape_original_width, opt.cityscape_original_height))
            # threasholding, the label of pedestrian is 24
            mask = resized_forward_result[:,:] > opt.threshold
            mask = mask.astype(float) * 24
            cv2.imwrite(forward_file_name, mask)
            # write a visualization of three images stitched together
            stitch_image = np.zeros([opt.cityscape_original_height*3,opt.cityscape_original_width,3])
            for j in range(3):
                stitch_image[0:opt.cityscape_original_height, 0:opt.cityscape_original_width,j] = resized_forward_result
                stitch_image[opt.cityscape_original_height:2*opt.cityscape_original_height, 0:opt.cityscape_original_width, j] = mask

            stitch_image[opt.cityscape_original_height*2:opt.cityscape_original_height*3, 0:opt.cityscape_original_width/2,:] = cv2.resize(np.reshape(original_image_batch[2*j,:,:,:]*128+128,[opt.input_h,opt.input_w,3]), (opt.cityscape_original_width/2, opt.cityscape_original_height))

            stitch_image[opt.cityscape_original_height*2:opt.cityscape_original_height*3, opt.cityscape_original_width/2:opt.cityscape_original_width,:] = cv2.resize(np.reshape(original_image_batch[2*j+1,:,:,:]*128+128,[opt.input_h,opt.input_w,3]), (opt.cityscape_original_width/2, opt.cityscape_original_height))

            cv2.imwrite(forward_file_name + "_stich.png", stitch_image)
        pass 

    # run the given evaluation code
    def get_eval_set_result(self, epoch_minus_one):
        os.system('python ./cityscapesScripts-master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py > logs/epoch' + str(epoch_minus_one+1) + '.txt')
        pass
    
    def reset(self):
        self.nowTestIdx = 0
        # for storing model
        if not os.path.exists(opt.test_results_cityscape_dir):
            os.makedirs(opt.test_results_cityscape_dir)
        else:
            shutil.rmtree(opt.test_results_cityscape_dir)
            os.makedirs(opt.test_results_cityscape_dir)


if __name__ == "__main__":
    pass
