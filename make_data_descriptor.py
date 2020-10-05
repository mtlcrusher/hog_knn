from os import listdir
from os.path import isfile, join

from skimage import feature
import cv2
import numpy as np

def make_data_desc(filenames, data_class):
    desc = []
    for filename in filenames:
        # print(filename)
        img = cv2.imread(data_class+filename)
        desc.append(feature.hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L1', visualize=False))
    
    return np.array(desc)

def build_dataset():
    pos_files = [f for f in listdir('dataset/positives/') if isfile(join('dataset/positives/', f))]
    pos_img_desc = make_data_desc(pos_files, "dataset/positives/")
    pos_target = np.full((pos_img_desc.shape[0], 1), 0)

    neg_files = [f for f in listdir('dataset/negatives/') if isfile(join('dataset/negatives/', f))]
    neg_img_desc = make_data_desc(neg_files, "dataset/negatives/")
    neg_target = np.full((neg_img_desc.shape[0], 1), 1)

    ran_files = [f for f in listdir('dataset/random/') if isfile(join('dataset/random/', f))]
    ran_img_desc = make_data_desc(ran_files, "dataset/random/")
    ran_target = np.full((ran_img_desc.shape[0], 1), 2)

    dataset = np.concatenate((pos_img_desc, neg_img_desc, ran_img_desc), axis=0)
    target = np.concatenate((pos_target, neg_target, ran_target), axis=0)

    return (dataset, target)
