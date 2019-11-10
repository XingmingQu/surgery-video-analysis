
# coding: utf-8

# In[31]:

print("Import an awesome Label helper")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# In[32]:

class Label:
    def __init__(self, label_csv_file):
        self.label_csv_file = pd.read_csv(label_csv_file)        
        self.r_max,self.r_min=self._set_rmax_and_rmin()
        self.trainee_dict=self._set_trainee_dict()

    def _set_rmax_and_rmin(self):  
        r_group=self.label_csv_file.groupby(['VideoNum'])
        r_max=r_group.max()
        r_min=r_group.min()
        r_max=r_max.reset_index()
        r_min=r_min.reset_index()
        return r_max,r_min
    
    def _set_trainee_dict(self):
        name_group=self.label_csv_file.groupby(['lastName'])
        return dict(name_group.VideoNum.unique())
        
    def video_count(self):
        counts=self.label_csv_file['VideoNum'].value_counts()
        return len(counts)
    
    def get_review_times_by_video_number(self,videoNum):
        counts=self.label_csv_file['VideoNum'].value_counts()
        times=int(counts[counts.index==videoNum])
        return times
    
    def get_video_score_range_by_video_number(self,videoNum):

        vmax=self.r_max[self.r_max['VideoNum']==videoNum]
        vmin=self.r_min[self.r_min['VideoNum']==videoNum]
        return vmax,vmin
    
    def get_video_mean_label_by_video_number(self,videoNum):
        group=self.label_csv_file[self.label_csv_file['VideoNum'] == videoNum]
        return np.array(group.mean()[1:])

    def get_video_median_label_by_video_number(self,videoNum):
        group=self.label_csv_file[self.label_csv_file['VideoNum'] == videoNum]
        return np.array(group.median()[1:])
    
    def get_video_level_by_video_number(self,videoNum,meanOrMedian):
        if meanOrMedian =="mean":
            group=self.label_csv_file[self.label_csv_file['VideoNum'] == videoNum]
            group=np.array(group.mean()[1:])
        else:
            group=self.label_csv_file[self.label_csv_file['VideoNum'] == videoNum]
            group=np.array(group.median()[1:])      
            
        totalScore= np.sum(group)
        
        if totalScore<=13:
            return 0,totalScore
        elif totalScore<=21:
            return 1,totalScore
        else:
            return 2,totalScore
        # GEARS 6-13 = novice=0, GEARS 14-21 intermediate=1, GEARS 22-30 expert=2


    def get_video_number(self):
        return self.label_csv_file.VideoNum.unique()
    
    def get_trainee_info(self):
        names=self.label_csv_file.lastName.unique()
        print("There are %d trainees in total"%len(names))      
        print("\n Number of Videos from each person")
        peoples=list(self.trainee_dict.keys())
        for people in peoples:
        	video_number_list=self.get_video_number_by_name(people)
        	print('%-15s' % people,"--",video_number_list,"----:",len(video_number_list))
        
    def get_video_number_by_name(self,name):
        return self.trainee_dict[name]
        
    def train_test_split_on_people(self,test_ratio):
        peoples=list(self.trainee_dict.keys())
        train_people,test_people= train_test_split(peoples,test_size=test_ratio)
        
        print("train on videos from:\n",train_people)
        print("test on videos from:\n",test_people)
        
        train_video_number=np.concatenate([self.get_video_number_by_name(people) for people in train_people], axis=None)
        test_video_number=np.concatenate([self.get_video_number_by_name(people) for people in test_people], axis=None)
        return train_video_number,test_video_number
    
    def cross_validation_on_people(self,folds):

        peoples=list(self.trainee_dict.keys())
        kf = KFold(n_splits=folds)
        train_people=[]
        test_people=[]
        for train_index, test_index in kf.split(peoples):
            train_people.append([peoples[i] for i in train_index ])
            test_people.append([peoples[i] for i in test_index ])

        train_video_number_folds=[]
        test_video_number_folds=[]

        for one_fold_train, one_fold_test in zip(train_people,test_people):
            train_video_number=np.concatenate([self.get_video_number_by_name(people) for people in one_fold_train], axis=None)
            test_video_number=np.concatenate([self.get_video_number_by_name(people) for people in one_fold_test], axis=None)

            train_video_number_folds.append(train_video_number)
            test_video_number_folds.append(test_video_number)


        return train_video_number_folds,test_video_number_folds

if __name__ == "__main__": 
    print("test...")       



# ll=Label('..//2019_fall_labels.csv')
# N=ll.get_video_number()



# a,b=ll.get_video_Scorerange(11)
# ll.get_video_median_label(11)

