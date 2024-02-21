import os
import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models 
from PIL import Image
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import keras
classes=['0_Compulsory Keep BothSide',
 '10_Maximum Speed 30',
 '11_Maximum Speed 40',
 '12_Maximum Speed 50',
 '13_Maximum Speed 60',
 '14_Maximum Speed 70',
 '15_Maximum Speed 80',
 '16_Maximum Speed 90',
 '17_MotorCycle Prohibited',
 '18_No Entry',
 '19_No Horn',
 '1_Compulsory Keep Left',
 '20_NO Stopping',
 '21_NO Waiting',
 '22_One way Traffic',
 '23_Park',
 '24_Park Forbidden',
 '25_Pedestrain',
 '26_Pedestrian crossing',
 '27_Right Bend',
 '28_Right Margin',
 '29_Right Turn Prohibited',
 '2_Compulsory Keep Right',
 '30_Road Work',
 '31_Roundabouts',
 '32_School',
 '33_School Crossing',
 '34_Side Road Right',
 '35_Slow',
 '36_Speed Camera',
 '37_STOP',
 '38_Truck Prohibited',
 '39_Two Way Traffic',
 '3_Cycle crossing',
 '40_U-Turn',
 '41_U-Turn Allowed',
 '42_U-turn Prohibited',
 '4_Danger',
 '5_Give Way',
 '6_Hump',
 '7_Left Bend',
 '8_Left Margin',
 '9_Left Turn Prohibited']

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
model=load_model('Persian_Traffic_Sign_V1-2.h5')
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224,224,3))
    #plt.imshow(img)
    #plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    print("Actual: "+(image_path.split("\\")[-1]).split("_")[0])
    print("Predicted: "+classes[np.argmax(pred)])
predict_image('test.jpg')
