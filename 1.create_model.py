#1.Import the libraries
import cv2 
import numpy as np
import os
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import models 
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


#2 tải dữ liệu ảnh dùng để train
TRAIN_DATA = 'DATA FACE DETECTION/TRAIN'
TEST_DATA = 'DATA FACE DETECTION/TEST'

#4 Khởi tạo dưc liệu và tạo One Hot Encoding
Xtrain = []
ytrain = []



Xtest = []
ytest = []

#One Hot Encoding
dict = {'DOAN THANH NAM':[1,0,0,0,0],
        'LE VAN QUANG':[0,1,0,0,0],
        'NGUYEN QUOC TIEN':[0,0,1,0,0],
        'TRAN LE NHAT HUY':[0,0,0,1,0],
        'VU DUC BINH':[0,0,0,0,1],
        'DOAN THANH NAM test':[1,0,0,0,0],
        'LE VAN QUANG test':[0,1,0,0,0],
        'NGUYEN QUOC TIEN test':[0,0,1,0,0],
        'TRAN LE NHAT HUY test':[0,0,0,1,0],
        'VU DUC BINH test':[0,0,0,0,1]
        }


#4 Xây dựng hàm lấy dữ liệu
def getData(dirData,lstData):
    for whatever in os.listdir(dirData):
        whatever_path = os.path.join(dirData,whatever)# duong dan di vao dataset
        #list trung gian luu ket qua cua vong lap thu 2
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path,filename)
            
            label = filename_path.split('\\')[1]
            img = np.array(Image.open(filename_path))
            lst_filename_path.append((img,dict[label]))

        lstData.extend(lst_filename_path)#list du lieu
    return lstData


Xtrain = getData(TRAIN_DATA, Xtrain)
Xtest = getData(TEST_DATA, Xtest)

np.random.shuffle(Xtrain)
np.random.shuffle(Xtrain)
np.random.shuffle(Xtrain)

np.random.shuffle(Xtest)
np.random.shuffle(Xtest)
np.random.shuffle(Xtest)
#print(Xtrain[10])

#5.xây dựng Model


model_training_first = models.Sequential([
    layers.Conv2D(32,(3,3),input_shape=(300,300,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.1),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(6, activation='softmax'),

])

model_training_first.summary()
'''
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.1,
    rescale=1./255,
	width_shift_range=0.1,height_shift_range=0.1,
	horizontal_flip=True,
    brightness_range=[0.2,1.5], fill_mode="nearest")
aug_val = ImageDataGenerator(rescale=1./255)
'''

model_training_first.compile(optimizer='Adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
model_training_first.fit(np.array([x[0] for i,x in enumerate(Xtrain)]), np.array([y[1] for i,y in enumerate(Xtrain)]) ,epochs=10)
                         
                               
#model_training_first.save('FACE_DETECTION(2).h5')



        