
# coding: utf-8

# In[1]:

print("import an awesome feature helper")
import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
import modules.labelHelper as lh 
from sklearn.model_selection import train_test_split
import random


# In[2]:


def get_raw_feature_by_folder_and_videoNumbers(video_data_folder,video_number):
    feature=[]
    #print("load feature from files, there are %d videos"%len(video_number))
    for number in video_number:
        file_name = "raw_{}.npy".format(str(number))
        each_video= pjoin(video_data_folder,file_name)
        try:
            feature.append(np.load(each_video))
        except:
            print("Video %d did not find" %number)
            video_number = video_number[video_number!=number]
    
    #print("after deop: ",len(video_number))
    return feature,video_number

def get_diff_and_hstack_to_orginal_data(X,time_lag=2,move_threshold=200):
    original=X[:-time_lag]
    modified=X[time_lag:]
    result=modified-original
    ## threshold
    # consider there was no top in the first image and it showed up in the next image
    # the difference would be huge, which was not ideal.
    # so we need to filter these extrem value 
    result[np.abs(result)>move_threshold]=0
    
    return np.hstack((X[time_lag:],result))

##from each video sample video clips with size=window_L. you can specify stride 
def make_video_clips(matrix,window_L,stride):
    alldata=[]
    total_frame=matrix.shape[0]
    index=[n for n in range(1,total_frame,stride)]
    for start_index in index:
        if start_index+window_L> total_frame:
            break
#         print(start_index)
        each_clip_data=matrix[start_index:start_index+window_L]
#         each_clip_data=np.transpose(each_clip_data)
#         print(each_clip_data.shape)
        alldata.append(each_clip_data)
    return np.array(alldata)

def make_video_feature_list(raw_feature,video_clips_length,time_lag,move_threshold,stride):
    training_data=[]
    for each_video in raw_feature:
        new_feature=get_diff_and_hstack_to_orginal_data(each_video,time_lag,move_threshold)
        video_clip=make_video_clips(new_feature,video_clips_length,stride)
#         print(video_clip.shape)
        training_data.append((video_clip))
    return np.array(training_data)


def stack_videoClips_and_getLabelByVideoNumber(training_data_list,video_number,label):
    if len(training_data_list)!= len(video_number):
        print('the length of training_data_list is not equal to video_number_list!')
        return 
    final_data=training_data_list[0]
    final_label=[label.get_video_mean_label_by_video_number(video_number[0]) for _ in range(final_data.shape[0])]

    for i in range(1,len(training_data_list)):
        final_data=np.vstack((final_data,training_data_list[i]))
        for j in range(training_data_list[i].shape[0]):
            final_label.append(label.get_video_mean_label_by_video_number(video_number[i]))
    return final_data,np.array(final_label)


def train_test_split_by_videoNumber(total_data,video_number,test_ratio,label):
    index=np.array(range(0,len(video_number)))    
    train_video_index,test_video_index= train_test_split(index,test_size=test_ratio)
    
    train_video=total_data[train_video_index]
    test_video=total_data[test_video_index]
    
    train_video_number=video_number[train_video_index]
    test_video_number=video_number[test_video_index]
    
    train_video,train_video_label=stack_videoClips_and_getLabelByVideoNumber(train_video,train_video_number,label)
    test_video,test_video_label=stack_videoClips_and_getLabelByVideoNumber(test_video,test_video_number,label)
    return train_video,train_video_label,test_video,test_video_label

def make_train_test_data_split_by_video_ratio(folder,label,video_clips_length,time_lag,move_threshold,stride,test_ratio):

    video_number = label.get_video_number()
#     print(video_number)
    print("before deop we have %d videos"%len(video_number))
    raw_feature,video_number=get_raw_feature_by_folder_and_videoNumbers(folder,video_number)
    print("we have %d video after drop"%len(video_number))
    training_data_list=make_video_feature_list(raw_feature,video_clips_length,time_lag,move_threshold,stride)
    return train_test_split_by_videoNumber(training_data_list,video_number,test_ratio,label)

def make_train_test_data_from_video_numbers(folder,video_number,video_clips_length,time_lag,move_threshold,stride,label):
    raw_feature,video_number=get_raw_feature_by_folder_and_videoNumbers(folder,video_number)
    data_list=make_video_feature_list(raw_feature,video_clips_length,time_lag,move_threshold,stride)
    video,video_label=stack_videoClips_and_getLabelByVideoNumber(data_list,video_number,label)
    return video,video_label


if __name__ == "__main__": 
    print("test...")     


# ll=lh.Label('..//2019_fall_labels.csv')
# video_number = ll.get_video_number()
# print(video_number)
# print("before deop: ",len(video_number))


# In[4]:


# folder = "C:\\2019_fall_video_features"

# raw_feature,video_number=get_raw_feature_by_folder_and_videoNumbers(folder,video_number)
# print("after deop: ",len(video_number))
# len(raw_feature)


# In[5]:


# ####################################### set video clips parameters
# ############################################### make new 28-d feature
# video_clips_length=30
# time_lag=2
# move_threshold=150
# stride=video_clips_length

# training_data_list=prepare_training_data_list(raw_feature,video_clips_length,time_lag,move_threshold,stride)


# In[6]:


# # x,y = stack_videoClips_and_getLabelByVideoNumber(training_data_list,video_number,ll)
# train_x,train_y,test_x,test_y=train_test_split_by_videoNumber(training_data_list,video_number,0.2,ll)

