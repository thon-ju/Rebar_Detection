import tensorflow as tf
import os

print("TensorFlow version:", tf.__version__)

files_array = []
filePath = 'D:\\ml-workspace\\Rebar_Detection\\data\\dataset\\train_data_VOC\\Annotations'
for file_name in os.listdir(filePath):
    files_array.append(os.path.splitext(file_name)[0])
print(files_array)