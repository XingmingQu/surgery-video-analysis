from matplotlib import pyplot as plt
import warnings
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.regularizers import l2 
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Dropout, Activation, Flatten,Conv1D, GlobalAveragePooling1D, MaxPooling1D,AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D,Conv1D,GlobalMaxPooling1D,MaxPooling1D,average, concatenate,RepeatVector,Lambda,add,subtract,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Input, Model
from sklearn import metrics as mt
from skimage.io import imshow
from sklearn import preprocessing
from matplotlib.patches import Rectangle
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)
warnings.filterwarnings('ignore')



def conv_blocks(ft_number,k_size,input_tensor):
    x = Conv1D(filters=ft_number, 
                     kernel_size=k_size, 
                     padding='same',
                     kernel_regularizer=l2(0.001)
              )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def op(inputs):
    x, y = inputs
    return K.pow((x - y), 2) 

def conv_branch(init_input,kernel_size):
    x=conv_blocks(ft_number=4,k_size=kernel_size,input_tensor=init_input)
    x=MaxPooling1D(2,padding='same')(x)

    x=conv_blocks(ft_number=8,k_size=kernel_size,input_tensor=x)
    x=MaxPooling1D(2,padding='same')(x)

    x=conv_blocks(ft_number=16,k_size=kernel_size,input_tensor=x)
    u = GlobalMaxPooling1D()(x)
    u_broadcast=RepeatVector(x.shape[1])(u)

    o=Lambda(op)([u_broadcast,x])  # K.pow((x - y), 2) 
    var = GlobalAveragePooling1D()(o)
    X_vector = concatenate([u,var])
    return X_vector


def multi_task_branch(input_feature,no_of_neuron_indense,dp_rate,branch_name):
    y1 = Dense(no_of_neuron_indense,kernel_initializer='he_uniform')(input_feature)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Dropout(dp_rate)(y1)
    
    y1= Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',name=branch_name)(y1)
    return y1

def make_model(l2_lambda,clip_lenth,dimension):
    init_input = Input(shape=(clip_lenth, dimension))
    
    #branch 1
    x=conv_branch(init_input,5)

    #branch 2
    y=conv_branch(init_input,8)
    
    #branch 3
    z=conv_branch(init_input,10)
    
    video_clip_feature = concatenate([x,y,z], name='video_clip_feature')
    
    no_of_neuron_indense=16
    dp_rate=0.3
    y1=multi_task_branch(video_clip_feature,no_of_neuron_indense,dp_rate, "DP")
    y2=multi_task_branch(video_clip_feature,no_of_neuron_indense,dp_rate, "BD")
    y3=multi_task_branch(video_clip_feature,no_of_neuron_indense,dp_rate, "E")
    y4=multi_task_branch(video_clip_feature,no_of_neuron_indense,dp_rate, "FS")
    y5=multi_task_branch(video_clip_feature,no_of_neuron_indense,dp_rate, "A")
    y6=multi_task_branch(video_clip_feature,no_of_neuron_indense,dp_rate, "RC")
    
    model = Model(inputs=init_input,outputs=[y1,y2,y3,y4,y5,y6])
#loss='mean_squared_error', # 'categorical_crossentropy' 'mean_squared_error' 'mean_absolute_percentage_error'
    losses = {
        "DP": "mean_squared_error",
        "BD": "mean_squared_error",
        "E": "mean_squared_error",
        "FS": "mean_squared_error",
        "A": "mean_squared_error",
        "RC": "mean_squared_error",
    }
    model.compile(loss=losses, optimizer='adam') # 'adadelta' 'rmsprop'                  

    return model    


def get_callback_list_by_model(model_name):
 
    checkpoint = ModelCheckpoint('model_zoo/'+model_name+'.h5', monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only = True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                       verbose=1, mode='auto', epsilon=0.0001)
    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=7)
    csv_logger = CSVLogger(filename='./training_log_'+model_name+'.csv',
                       separator=',',
                       append=True)
    
    callbacks_list = [checkpoint,reduceLROnPlat, early,csv_logger]
    return callbacks_list



def generator(data, label, batch_size=64,noise_range=0.1):
    while 1:
        DP = []  
        BD = []
        E = []
        FS = []
        A = []
        RC = []
        rows = np.random.randint(0, data.shape[0], size=batch_size)

        samples = data[rows]
        y=label[rows]
        if noise_range != 0:
            noise=np.random.uniform(low=-noise_range, high=noise_range, size=(y.shape[0],y.shape[1]))
            y=y+noise
            
        for i in range(len(y)):
            DP.append(y[i][0])
            BD.append(y[i][1])
            E.append(y[i][2])
            FS.append(y[i][3])
            A.append(y[i][4])
            RC.append(y[i][5])
        labels = [np.array(DP),np.array(BD),np.array(E),np.array(FS), np.array(A),np.array(RC)]
        yield samples,labels