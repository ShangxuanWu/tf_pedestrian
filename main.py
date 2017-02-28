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
import timeit

# local import files
import opt
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
        #self.train_tfrecords = [opt.train_tfrecord_path + 'train_part' + str(i) for i in range(10) + '.tfrecord']

    def reset(self):
        # reset the index to 0
        self.nowTestIdx = 0
        self.nowTrainIdx = 0    
    
    def get_next_train_batch(self):
        #with tf.device('/cpu:0'):
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
            #gt_fn = img_fn.replace('image', 'gt')[:-4]+'.mat'
            gt_fn = img_fn.replace('donghyun', 'shangxuan')[:-4]+'.mat'
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

# the helper function for reading tfrecord for input()
def read_and_decode(filename_queue):

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.string),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.float64)
  image.set_shape([opt.input_h*opt.input_w*3])
  #pdb.set_trace()
  image_ = tf.reshape(image, [opt.input_h, opt.input_w,3])
  image__ = tf.cast(image_, tf.float32)

  label = tf.decode_raw(features['label'], tf.uint8)
  label.set_shape([opt.input_h*opt.input_w*opt.nOutputs])
  label_ = tf.reshape(label, [opt.input_h, opt.input_w,opt.nOutputs])
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  #image = tf.cast(image, tf.float32)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  #label = tf.cast(features['label'], tf.int32)

  return image__, label_


# reading tfrecords
def inputs():
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(opt.train_tfrecord_fns, num_epochs=opt.epoch)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=opt.train_batch_size, num_threads=2,
        capacity=1000 + 3 * opt.train_batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels


# main function
if __name__ == '__main__':
    
    with tf.Graph().as_default():
        model = model_3stage.cnn_model()

        # set up input and output
        #x, gt = model.setup_input()
        images, labels = inputs()
        
        # build model
        final_stage_output_deconved, loss, train_op = model.setup_model(images, labels)
        
        # new data fetch and save object
        #data_fetch_obj = read_data()
        data_save_obj = save_result()
        cityscape = test_cityscape.cityscape_data()
        
        # start session
        sess = tf.Session()
        
           
        # start logger using matplotlib
        loss_record_stack = []

        # initialize all
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        # new tensorflow model saver object
        saver = tf.train.Saver()
        
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # restore variables
        if opt.load_pretrain:
            saver.restore(sess, opt.pretrain_dir)
            print("Model restored.")
        
        # start training and testing
        #for i in range(opt.epoch):    
            # training
            #data_fetch_obj.reset()
            #for j in range(int(math.floor(opt.total_image_num / opt.train_batch_size))):
        try:
            step = 0
            while not coord.should_stop():    
                # read
                #start = timeit.default_timer()
                #batch_x, batch_y = data_fetch_obj.get_next_train_batch()
                #end = timeit.default_timer()
                #print( "data loading time %f " %(end-start))
                #
                #start = timeit.default_timer()
                #this_batch_loss = loss.eval(feed_dict={x: batch_x, gt: batch_y})
                
                step += 1
                start = timeit.default_timer()
                _, this_batch_loss = sess.run([train_op, loss])
                end = timeit.default_timer()
                print( "one batch training time %f " %(end-start))
                # logger
                loss_record_stack.append(this_batch_loss)
                # print loss
                print('Batch %d: loss %f' % (step, this_batch_loss))
                
                # back-propagation
                #train_step.run(feed_dict={x: batch_x, gt: batch_y})
                #end = timeit.default_timer()
                #print( "data forwarding time %f " %(end-start))

                if step % opt.draw_loss_interval == 0:
                    # save plot for training loss
                    loss_np_array = np.array(loss_record_stack)
                    plt.plot(loss_np_array)
                    plt.savefig(opt.loss_plot_path)
                    print('Loss plotted !')
                
                # you can check particular tensor in pdb: "print(tf.get_default_graph().get_tensor_by_name("stage1/W_2:0").eval())", if checking the feature map, should feed_dict in the eval() 
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (opt.epoch, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            # testing (forwarding the selected 20 testing images to visualize the result)
            '''cityscape.reset()
            for j in range(100): # 5 images per batch, 500 validation images in cityscape val set
                print('Testing Batch %d' % (j+1))
                batch_x, batch_y = cityscape.get_next_test_batch()
                batch_image_forward_result = final_stage_output_deconved.eval(feed_dict={x: batch_x, gt: batch_y})

                # save forward results
                cityscape.save_forward_result(batch_image_forward_result, j)'''
                
        # save the model for this epoch
        save_path = saver.save(sess, opt.model_save_path)
        print('Batch %d model saved !' % (step))

        '''    # test cityscape for formal accuracy
            cityscape.get_eval_set_result(i)
            print('Got IoU of CityScape!')'''
        
        #coord.request_stop()
        pdb.set_trace()
        print('Optimize Finished. Training Data Logged.')
        coord.join(threads)
        sess.close()
