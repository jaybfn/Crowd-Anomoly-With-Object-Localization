# importing libraries

import numpy as np
import cv2
from PIL import Image as im
from PIL import Image
import os
from os import listdir
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MinMaxScaler

# image to numpy transformation
image_gray = []
def data_preprocessing(image):
    
    img=img_to_array(image)
    img=resize(img,(227,227,3))
    gray_img=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]
    gray_img_scaled = scaler.fit_transform(gray_img)
    image_gray.append(gray_img_scaled)
    return image_gray

if __name__ == '__main__':

    # data normalization
    scaler = MinMaxScaler()

    # path to the images
    src_path_train = "../data/ucf_crime/training_mini/Normal/"
    src_path_val = "../data/ucf_crime/validation_mini/Normal/"
    src_path_test = "../data/ucf_crime/testing_mini/Anomaly/"

    # image to numpy transformation
    #folder_dir = [src_path_train, src_path_val, src_path_test]
    #file_names = ['train.npy', 'validation.npy', 'test.npy']

    folder_dir = [src_path_train, src_path_val, src_path_test]
    file_names = ['normal_mini_tr.npy', 'normal_mini_val.npy','Anomaly_mini_test.npy']
    for dir, names in zip(folder_dir, file_names):
        for images in os.listdir(dir):
            img = Image.open(dir + images)
            image_gray = data_preprocessing(img)
        gray_npimg = np.array(image_gray)
        nr_img, x_size, y_size = gray_npimg.shape
        gray_npimg.resize(x_size, y_size,nr_img)
        gray_npimg=np.clip(gray_npimg,0,1)
        np.save('../data/'+names, gray_npimg)



