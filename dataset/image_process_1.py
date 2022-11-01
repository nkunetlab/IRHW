'''
Process source dataset of Intel images and heritage, for the input of IRN or image classifier
'''
import os
import cv2

# destination trainset, testset of Intel images, heritage, as input of IRN
dst_image_path = "./heritage_images_128_128_png/"
# dst_image_path = "./Intel_images_images_150_150_png/"
if not os.path.exists(dst_image_path):
    os.mkdir(dst_image_path)

# destination trainset, testset of Intel images, heritage, as input of image classifier
dst_train_path = "./heritage_128_128_png/train/"
# dst_train_path = "./Intel_images_150_150_png/train/"
dst_test_path = "./heritage_128_128_png/test/"
# dst_test_path = "./Intel_images_150_150_png/test/"
if not os.path.exists(dst_train_path):
    os.mkdir(dst_train_path)
if not os.path.exists(dst_test_path):
    os.mkdir(dst_test_path)

# source trainset, testset of Intel images, heritage
src_train_path = "./heritage/train/"
# src_train_path = "./Intel_images/train/"
src_test_path = "./heritage/test/"
# src_test_path = "./Intel_images/test/"

def process(path, dst_path):
    folders = os.listdir(path)
    for folder in folders:
        next_folder = path + folder + "/"
        dst_next_folder = dst_path + folder + "/"
        if not os.path.exists(dst_next_folder):
            os.mkdir(dst_next_folder)
        for file in os.listdir(next_folder):
            file_path = next_folder + file
            dst_img_path = dst_next_folder + file
            dst_file_path = dst_image_path + file
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))
            dst_img_path = dst_img_path.replace(".jpg", ".png")
            dst_file_path = dst_file_path.replace(".jpg", ".png")
            cv2.imwrite(dst_img_path, img)
            cv2.imwrite(dst_file_path, img)

process(src_train_path, dst_train_path)
process(src_test_path, dst_test_path)