import glob
import os 
from os import listdir
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from PIL import Image as im
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import random
from datetime import datetime 
from numpy.random import seed
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
plt.style.use('fivethirtyeight')

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose


def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

def reshape_array(numpy_data_X, BATCH_SIZE):
    
    frames=numpy_data_X.shape[2]
    frames=frames-frames%BATCH_SIZE
    numpy_data_X=numpy_data_X[:,:,:frames]
    numpy_data_X=numpy_data_X.reshape(-1,SIZE,SIZE,BATCH_SIZE)
    numpy_data_X=np.expand_dims(numpy_data_X,axis=4)
    numpy_data_y=numpy_data_X.copy()
    return numpy_data_X, numpy_data_y

def img_transformation(generators):
    """ for 3D conv we need an extra dimention in the data"""
    x ,y = generators.__next__()
    x = np.expand_dims(x,axis=4)
    y = x.copy()
    return x ,y

def metricplot(df, xlab, ylab_1,ylab_2, path):
    
    """
    This function plots metric curves and saves it
    to respective folder
    inputs: df : pandas dataframe 
            xlab: x-axis
            ylab_1 : yaxis_1
            ylab_2 : yaxis_2
            path: full path for saving the plot
            """
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(x = df[xlab], y = df[ylab_1])
    sns.lineplot(x = df[xlab], y = df[ylab_2])
    plt.xlabel('Epochs',fontsize = 12)
    plt.ylabel(ylab_1,fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend([ylab_1,ylab_2], prop={"size":12})
    plt.savefig(path+'/'+ ylab_1)
    #plt.show()
def load_model():
    """
Return the model used for abnormal event 
detection in videos using spatiotemporal autoencoder

"""
    model = Sequential()

    model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
    model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))



    model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))


    model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))


    model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))




    model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
    model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

    return model


if __name__=='__main__':

    with tf.device('/gpu:0'):
        seed(42)
        tf.random.set_seed(42) 
        keras.backend.clear_session()

        # gpus = tf.config.list_physical_devices('GPU')

        # if gpus:

        #     try:
                
        #         # Currently, memory growth needs to be the same across GPUs
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #         logical_gpus = tf.config.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #     except RuntimeError as e:
        #         # Memory growth must be set before GPUs have been initialized
        #         print(e)
        


        #creating main folder
        today = datetime.now()
        today  = today.strftime('%Y_%m_%d')
        path = 'Model_Outputs/'+ today
        create_dir(path)

        # creating directory to save model and its output
        EXPERIMENT_NAME = input('Enter new Experiment name:')
        print('\n')
        print('A folder with',EXPERIMENT_NAME,'name has be created to store all the model details!')
        print('\n')
        folder = EXPERIMENT_NAME
        path_main = path + '/'+ folder
        create_dir(path_main)

        # creating directory to save all the metric data
        folder = 'metrics'
        path_metrics = path_main +'/'+ folder
        create_dir(path_metrics)

        # creating folder to save model.h5 file
        folder = 'model'
        path_model = path_main +'/'+ folder
        create_dir(path_model)

        # creating folder to save model.h5 file
        folder = 'model_checkpoint'
        path_checkpoint = path_main +'/'+ folder
        create_dir(path_checkpoint) 


        # image_size 
        SIZE = 227
        # model parameters
        FILTERS = 128
        ACTIVATION = 'tanh'
        BATCH_SIZE = 10
        EPOCHS = 200

        # model name
        model_name = 'model.h5'

        #path for the image dataset
        src_path_train = "../data/ucf_crime/normal_mini_tr.npy"
        src_path_val = "../data/ucf_crime/normal_mini_val.npy"
        #src_path_test = "data/Anomaly_test.npy"

        X_data=np.load(src_path_train)

        #X_test=np.load(src_path_test)

        #X_train, X_val = train_test_split(X_data, test_size=0.25, random_state=42)



        #X_test, y_test = reshape_array(X_test)

        X_train, y_train = reshape_array(X_data, BATCH_SIZE)
        print(X_train.shape)

        X_val=np.load(src_path_val)

        X_val, y_val = reshape_array(X_val, BATCH_SIZE)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

        BATCH_SIZE_NEW = 10
        #SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.batch(BATCH_SIZE_NEW)
        val_dataset = test_dataset.batch(BATCH_SIZE_NEW)
        
        # load the model
        Model = load_model()
        Model.summary()

        Model.compile(optimizer=keras.optimizers.Adam(0.001), 
                    loss='mean_squared_error',metrics=['accuracy'])

        cb = [
            tf.keras.callbacks.ModelCheckpoint(path_model+'/'+model_name),
            tf.keras.callbacks.ModelCheckpoint(path_checkpoint),
            tf.keras.callbacks.CSVLogger(path_metrics+'/'+'data.csv'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1001, restore_best_weights=False)] 

        history = Model.fit(train_dataset,
                batch_size = BATCH_SIZE, 
                epochs = EPOCHS,
                validation_data =  val_dataset,
                verbose = 1, 
                shuffle=True,
                callbacks=[cb])
        # 3D

    Model.save(path_model+'/'+'model.h5')

    #calculating losses!

    #train_loss, train_acc = Model.evaluate(X_train, y_train)
    #print('\n','Evaluation of Training dataset:','\n''\n','train_loss:',round(train_loss,3),'\n','train_acc:',round(train_acc,3),'\n')

    val_loss, val_acc = Model.evaluate(X_val, y_val)
    print('\n','Evaluation of Validation dataset:','\n''\n','val_loss:',round(val_loss,3),'\n','val_acc:',round(val_acc,3),'\n')

    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print('\n','Evaluation of Testing dataset:','\n''\n','test_loss:',round(test_loss,3),'\n','test_acc:',round(test_acc,3),'\n')

    # reading the data.csv where all the epoch training scores are stored
    df = pd.read_csv(path_metrics+'/'+'data.csv')   

    metricplot(df, 'epoch', 'loss','val_loss', path_metrics)
    metricplot(df, 'epoch', 'accuracy','val_accuracy', path_metrics)

