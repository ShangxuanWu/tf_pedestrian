# data source
train_img_list_txt = './file_lists/train_list.txt'
test_img_list_txt = './file_lists/cityscape_val.txt'
mean = [41,42,40]
std = [0.26,0.26,0.27]
total_image_num = 48000
#total_image_num = 12 # this is for testing
nOutputs = 15 # nJoint + 1
# 12 joint
#joint_names = ['segmentation', 'center', 'lower_neck', 'upper_neck', 'head', 'right_foot', 'left_foot', 'left_knee', 'right_knee', 'belly_button', 'left_hand', 'right_hand']
# 14 joint
joint_names = ['segmentation', 'Right_ankle', 'Right_knee', 'Right_hip', 'Left_hip', 'Left_knee', 'Left_ankle', 'Right_wrist', 'Right_elbow', 'Right_shoulder', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Neck_Head', 'top']

# data saving
save_dir = '/home/shangxuan/tf_pedestrian/results_epoch'
model_save_path = '/home/shangxuan/tf_pedestrian/saved_model/model.ckpt'
log_dir = '/home/shangxuan/tf_pedestrian/logs'
loss_plot_path = '/home/shangxuan/tf_pedestrian/loss.png'

# pre-trained model
pretrain_dir = './saved_model/model.ckpt'
load_pretrain = False

# original_CPM_model
original_cpm_model = './original_CPM/CPM-original.npy'
load_original_CPM = True

# train parameters
input_h = 368
input_w = 368
lr = 1e-4
epoch = 4
train_batch_size = 10
draw_loss_interval = 1000

# test parameters
test_image_batch = 10 # numbers of image to forward and save

# test cityscape parameters
test_img_list_txt_cityscape = './file_lists/cityscape_val.txt'
test_results_cityscape_dir = '/mnt/sdc1/shangxuan/dataset/cityscape/results/'
cityscape_original_height = 1024
cityscape_original_width = 2048
threshold = 70