import numpy as np
import os
import cv2
import pickle5 as pickle
from os.path import join as pjoin

def load_dict(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def visualization(fold_path, imageData_path, img_info, fixData_path):
    num = int(fold_path.split("_")[-1])
    img = img_info[num]
    pre = np.load(pjoin(fold_path, "results.npy"))
    pre = np.transpose(np.squeeze(pre))
    Mean = np.load("E:/1/Dataset/dataset/inputStandardization/mean_s1.npy")
    Std = np.load("E:/1/Dataset/dataset/inputStandardization/std_s1.npy")
    pre = pre * Std + Mean
    gt = np.load(pjoin(fixData_path, str(num)+".npy"))
    point_size = 5
    point_color = (0, 0, 255)
    thickness = 3
    pixel_width, pixel_height = 512, 256
    if not os.path.exists(pjoin(fold_path, 'pre_image')):
        os.makedirs(pjoin(fold_path, 'pre_image'))
    if not os.path.exists(pjoin(fold_path, 'raw_image')):
        os.makedirs(pjoin(fold_path, 'raw_image'))
    part_name = img["part"]
    image_id = list(img["image"])
    image_id.sort(key = lambda x: float(x[0:-4]))
    for i in range(5):
        image_path = pjoin(imageData_path, part_name, "panorama", image_id[i])
        img = cv2.imread(image_path)
        x = pre[i][0] * pixel_width
        y = (1 - pre[i][1]) * pixel_height
        cv2.circle(img, (int(x),int(y)), point_size, point_color, thickness)
        path = pjoin(fold_path, 'pre_image', image_id[i])
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        img = cv2.imread(image_path)
        x = gt[i][0] * pixel_width
        y = (1 - gt[i][1]) * pixel_height
        cv2.circle(img, (int(x),int(y)), point_size, point_color, thickness)
        path = pjoin(fold_path, 'raw_image',image_id[i])
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def main():
    fold_path = "path to sample folder"
    imageData_path = "E:/1/Dataset/rawData/rawImage/rawImage"
    img_info = load_dict("./dataset/image_info")
    fixData_path = "E:/1/Dataset/dataset/fix_data_stage1"
    visualization(fold_path, imageData_path, img_info, fixData_path)

if __name__ == "__main__":
    main()