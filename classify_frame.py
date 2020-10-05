import numpy as np
import cv2
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier

from make_data_descriptor import *

def make_test_data():
    img = cv2.imread('test/test-frame.png')
    
    window_w = 32
    window_h = 32
    
    cropped_img = img[int(0.25*img.shape[0]):int(0.75*img.shape[0])]

    img_desc = []
    window_img = []
    y_loc = 0
    while y_loc + window_h <= cropped_img.shape[0]:
        x_loc = 0
        while x_loc + window_w <= cropped_img.shape[1]:
            window = np.copy(cropped_img[y_loc : y_loc+window_h, x_loc : x_loc+window_w])
            window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            boxed_img = np.copy(cropped_img)
            boxed_img = cv2.rectangle(boxed_img, (x_loc, y_loc), (x_loc+window_w, y_loc+window_h), (0,255,0), 2)
            fdesc, HoGVis = feature.hog(window, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L1', visualize=True)
            HoGVis = exposure.rescale_intensity(HoGVis, in_range=(0, 10))
            cv2.imshow("image", img)
            cv2.imshow("resized", boxed_img)
            cv2.imshow("HoG", HoGVis)
            cv2.imshow("crop", window)
            x_loc += 8
            # print(type(fdesc))
            cv2.waitKey(1)
            img_desc.append(fdesc)
            window_img.append(window)
        y_loc += 8
    
    return (np.array(img_desc), np.array(window_img))

# build test data from scratch
test_desc, test_img = make_test_data()
np.savetxt("data_csv/test_data_desc.csv", test_desc, delimiter=",")

# build dataset from scratch
# dataset = build_dataset()
# np.savetxt("data_csv/data.csv", dataset[0], delimiter=",")
# np.savetxt("data_csv/target.csv", dataset[1], delimiter=",")

# read test data from file
test_desc = np.genfromtxt('data_csv/test_data_desc.csv', delimiter=',')
print(test_desc.shape)

# read dataset from file
data = np.genfromtxt('data_csv/data.csv', delimiter=',')
target = np.genfromtxt('data_csv/target.csv', delimiter=',')
print(data.shape)
print(target.shape)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance') 
knn.fit(data, target)

result = knn.predict(test_desc)

# write images and predictions
for i in range(len(test_img)):
    cv2.imwrite("{}{}{}".format("result/img/", i, ".png"), test_img[i])
np.savetxt("result/predction.csv", result)
