# helper functions
import cv2
import opts as opt

def read_txt(file_name):
    inputLines = []
    with open(file_name) as inputStream:
        for separateLine in iter(inputStream):
            inputLines.append(separateLine.strip('\n'))
    return inputLines

def preprocess_img(img):
    # resize
    img = cv2.resize(img, (opt.input_w, opt.input_h))
    # zero mean
    img = (img.astype(float) - 128)/128
    return img

# for splitting cityscape dataset images into half, and fill the forward blob
def preprocess_img_cut_half(full_img):
    # resize
    left_img = full_img[0:1023,128:895,:]
    right_img = full_img[1024:2047,128:895,:]
    left_img = cv2.resize(left_img, (opt.input_w, opt.input_h))
    right_img = cv2.resize(right_img, (opt.input_w, opt.input_h))
    # zero mean
    img = np.zeros([2,opt.input_h,opt.input_w,3])
    img[0,:,:,:] = left_img
    img[1,:,:,:] = right_img
    img = (img.astype(float) - 128)/128
    return img