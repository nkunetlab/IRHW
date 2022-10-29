'''
Make small dataset from watermarked testset for fine-tuning
'''
import os
import random
import cv2
import numpy as np

test_rate = 0.2
src_test = "./test/"
src_test_list = os.listdir(src_test)

if not os.path.exists("test_small/"):
    os.mkdir("test_small/")
if not os.path.exists("test_small/train/"):
    os.mkdir("test_small/train/")
if not os.path.exists("test_small/test/"):
    os.mkdir("test_small/test/")

def read_img(path):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def add_gauss_noise(image, mean=0, val=1e-25):
    size = image.shape
    image = image.astype(np.float32) / 255.
    gauss = np.random.normal(mean, val**0.05, size)
    image = image + gauss
    image = np.clip(image, 0, 1)
    image = (image * 255.0).round()
    return image.astype(np.uint8)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

for dir in src_test_list:
    class_dir = src_test + dir + "/"
    dst_class_dir_train = "./test_small/train/" + dir + "/"
    dst_class_dir_test = "./test_small/test/" + dir + "/"
    if not os.path.exists(dst_class_dir_train):
        os.mkdir(dst_class_dir_train)
    if not os.path.exists(dst_class_dir_test):
        os.mkdir(dst_class_dir_test)
    class_dir_list = os.listdir(class_dir)
    test_samples = random.sample(class_dir_list, int(len(class_dir_list) * test_rate))
    for file in class_dir_list:
        img = read_img(class_dir + file)
        img = add_gauss_noise(img)
        if file in test_samples:
            save_img(img, dst_class_dir_test + file)
        else:
            save_img(img, dst_class_dir_train + file)
