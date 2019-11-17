
# coding: utf-8

##Please put this script in the keras-yolo3 folder
# to run this script, please run:
# python -f C:\\New_video_images\\ -s C:\\2019_fall_video_features\\



# This notebook used trained yolo model to extract positions of robotic arms.
# 
# The results are 19 npy files which recorded the 14-D feature of each frame. The npy files are in the zip file "video feature".
# 
# Eg. video 1 has 800 frames and the feature will be (800,14).
# 
# So no need to run this notebook.

# In[1]:




import argparse
import json
from utils.utils import get_yolo_boxes, makedirs

import pandas as pd
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import cv2
import os
from os.path import join as pjoin
from matplotlib import pyplot as plt
import random
import warnings
import keras
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import average 
from keras.models import Input, Model
from sklearn import metrics as mt
from matplotlib import pyplot as plt
from skimage.io import imshow
warnings.filterwarnings('ignore')


# In[ ]:


parser = argparse.ArgumentParser(
  description='using YOLO to extract object locations from images')
parser.add_argument('-f', type=str, default='C:\\New_video_images\\',
                    help='root folder, there will be subfold inside of this folder')

parser.add_argument('-s', type=str, default='C:\\2019_fall_video_features\\',
                    help='output file folder')



args = parser.parse_args()


# In[2]:


### set up yolo
### load trained weights 
config_path  = 'config.json'


with open(config_path) as config_buffer:    
    config = json.load(config_buffer)

###############################
#   Set some yolo parameter
###############################       
net_h, net_w = 320, 320 # a multiple of 32, the smaller the faster
obj_thresh, nms_thresh = 0.65, 0.001   # nms_thresh should be set very small to prevent multiple bbox

###############################
#   Load the model
###############################
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
infer_model = load_model(config['train']['saved_weights_name'])


# In[3]:


def getbboxarr(img):
    ###############################
    #   Predict bounding boxes  and get the position
    ###############################
    image = cv2.imread(img)

    # predict the bounding boxes
    boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

    # draw bounding boxes on the image using labels
    _,arr=draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 
    return arr


# In[4]:


# find items from a list
# for exmaple our list might be [[rear_joint,100,200],[rear_joint,200,200],[front_joint,150,200],[front_joint,175,200] ]
# we want to know how many times the  rear_joint appeared 
def findAndReturn(List,name):
    count=0
    result=[]
    for i in range(len(List)):
        if List[i][0]==name:
            count=count+1
            result.append([List[i][1],List[i][2]])
    return count,result

# if we have two rear_joint in the list , we need to figure out
# which one was left and which one was right
# we can just x position 
def swap(List):
    if List[0][0]>List[1][0]:
        temp=List[0]
        List[0]=List[1]
        List[1]=temp
    return List

feature_names=['left_rear_joint_x','left_rear_joint_y','right_rear_joint_x','right_rear_joint_y',            'left_front_joint_x','left_front_joint_y','right_front_joint_x','right_front_joint_y',           'left_top_x','left_top_y','right_top_x','right_top_y', 'needle_x','needle_y'
           ]

# so we can go through the list and get feature from each image 
# 14 features was showed on above feature_names
# this function convert list to a vector 
# so each image will have its 14d features.
def checkLeftRightAndAssign(List):
    df_vector=pd.DataFrame(np.zeros(14)).transpose()
    df_vector.columns=feature_names
    arm=None
#check rear joint    
    count,position=findAndReturn(List,'rear_joint')
    if count==2:
        position=swap(position)
        df_vector.left_rear_joint_x=position[0][0]
        df_vector.left_rear_joint_y=position[0][1]
        df_vector.right_rear_joint_x=position[1][0]
        df_vector.right_rear_joint_y=position[1][1]
    if count==1:
        # add arm to save it is right arm or left arm
        # because rear_joint almost never exceed the half screen
        # so when there is only one arm, we can decide it is right arm or left arm
        if position[0][0]<320:
            df_vector.left_rear_joint_x=position[0][0]
            df_vector.left_rear_joint_y=position[0][1]
            arm=0
            # 0 for left
        else:
            df_vector.right_rear_joint_x=position[0][0]
            df_vector.right_rear_joint_y=position[0][1]  
            arm=1
            # 1 for right
#check front joint
    count,position=findAndReturn(List,'front_joint')
    if count==2:
            position=swap(position)
            df_vector.left_front_joint_x=position[0][0]
            df_vector.left_front_joint_y=position[0][1]
            df_vector.right_front_joint_x=position[1][0]
            df_vector.right_front_joint_y=position[1][1]
    if count==1:
        if position[0][0]<320:
            df_vector.left_front_joint_x=position[0][0]
            df_vector.left_front_joint_y=position[0][1]
        else:
            df_vector.right_front_joint_x=position[0][0]
            df_vector.right_front_joint_y=position[0][1]  
            
#check the top ***This doesn't account if they overlap, which they do overlap alot*** we need to discuss with Eric
    count,position=findAndReturn(List,'top')
    if count==2:
        position=swap(position)
        df_vector.left_top_x=position[0][0]
        df_vector.left_top_y=position[0][1]
        df_vector.right_top_x=position[1][0]
        df_vector.right_top_y=position[1][1]
    if count==1:
        if arm!= None:
            if arm == 0:
                df_vector.left_top_x=position[0][0]
                df_vector.left_top_y=position[0][1] 
            else:
                df_vector.right_top_x=position[0][0]
                df_vector.right_top_y=position[0][1]                  
        # if we do not have arm infomation, just use default
        else:
            if position[0][0]<320:
                df_vector.left_top_x=position[0][0]
                df_vector.left_top_y=position[0][1]
            else:
                df_vector.right_top_x=position[0][0]
                df_vector.right_top_y=position[0][1]  
            
#check the needle
    count,position=findAndReturn(List,'needle')
# to do if count>1
    if count==1:
        df_vector.needle_x=position[0][0]
        df_vector.needle_y=position[0][1]
    else:
        df_vector.needle_x=0
        df_vector.needle_y=0
    return df_vector   


# In[17]:


def readDataAndGetLabel(data_dir):
    Y=[]
    alldata=[]
    #data_dir contains all the sub folders. Named as 1 2 3 4 5 .......
    for folder in os.listdir(data_dir):
        # for each video
        each_data=[]   
        each_dir= pjoin(data_dir, folder)      #get each folder's name
        print('Now reading from folder.',each_dir, '----frames number=',len(os.listdir(each_dir)))
        Y.append(int(folder))
        
        # read image in the folder
        for i in range(1,len(os.listdir(each_dir))+1): 
            img_dir = each_dir+'\\'+ str(folder)+'_'+str(i)+'.jpg'
            readimg=checkLeftRightAndAssign(getbboxarr(img_dir))    # read each images
            readimg= np.squeeze(readimg)
            readimg=np.array(readimg,dtype='int') # we get(14,)

            each_data.append(readimg)
            
        alldata.append(np.array(each_data))
    return alldata,Y



# In[19]:


datadir=args.f
Data,Y=readDataAndGetLabel(datadir)


# In[9]:


for d,label in zip(Data,Y):
    np.save(args.s+"raw_"+str(label), d)

print("All the feature data saved in ",args.s)