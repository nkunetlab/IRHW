'''
Process the output of IRN, generate watermarked dataset to train image classifier
'''
import os

data_root = "../results/IRN_x2_watermark_heritage_2/set5/"
# data_root = "../results/IRN_x2_watermark_intel_26/set5/"
dst_root = "../heritage_images_watermark_heritage_2/"
# dst_root = "../Intel_images_images_watermark_intel_26/"

if not os.path.exists(dst_root):
    os.mkdir(dst_root)
all_files = os.listdir(data_root)
for file in all_files:
    suffix_1 = '_LR.png'
    suffix_2 = '_GT.png'
    suffix_3 = '_LR_ref.png'
    suffix_4 = '.pth'
    suffix_5 = '_diff_SR_mean.png'
    if suffix_1 in file or suffix_2 in file or suffix_3 in file or suffix_4 in file or suffix_5 in file:
        continue
    src_file_path = data_root + file
    dst_file_path = dst_root + file
    with open(src_file_path, "rb") as f1:
        with open(dst_file_path, "wb") as f2:
            f2.write(f1.read())

src_path = dst_root
dst_root_path = "./heritage_watermark_heritage_2/"
# dst_root_path = "./Intel_images_watermark_intel_26/"
if not os.path.exists(dst_root_path):
    os.mkdir(dst_root_path)

src_train_path = "./heritage/train/"
# src_train_path = "./Intel_images/train/"
src_test_path = "./heritage/test/"
# src_test_path = "./Intel_images/test/"

def process(path, dst_path):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    label_dict = {}
    folders = os.listdir(path)
    for folder in folders:
        next_folder = path + folder + "/"
        for file in os.listdir(next_folder):
            label_dict[file[:file.find(".jpg")]] = folder
    for file in os.listdir(src_path):
        try:
            label = label_dict[file[:file.find(".png")]]
            dst_folder = dst_path + label + "/"
            if not os.path.exists(dst_folder):
                os.mkdir(dst_folder)
            src_file = src_path + file
            dst_file = dst_folder + file
            with open(src_file, "rb") as f1:
                with open(dst_file, "wb") as f2:
                    f2.write(f1.read())
        except KeyError:
            pass


process(src_train_path, dst_root_path + "train/")
process(src_test_path, dst_root_path + "test/")
