import cv2
import numpy as np
import pandas as pd
from constants import img_dim, data_folder_path, number_of_classes
from PIL import Image
import keras

def read_and_rescale_image(img_path):
    img = cv2.imread(img_path)
    # this_im = Image.fromarray(img, 'RGB')
    # this_im.show()
    res = cv2.resize(img, dsize=(img_dim, img_dim), interpolation=cv2.INTER_CUBIC)
    # this_im = Image.fromarray(res, 'RGB')
    # this_im.show()
    return res

def read_image_files(csv_file_name):
    df = pd.read_csv(csv_file_name)
    labels, data = [], []
    for _, row in df.iterrows():
        this_items_lable = row['ClassId']
        this_items_path = data_folder_path+row['Path']
        this_img = read_and_rescale_image(this_items_path)
        
        labels.append(this_items_lable)
        data.append(this_img)
    labels, data = np.array(labels), np.array(data)
    labels = keras.utils.to_categorical(labels, num_classes=number_of_classes)
    return data, labels
