import os
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
import random
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten,Input
from tensorflow.keras import layers
import warnings

from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')

Train = os.path.join(os.getcwd(), "train")
Test = os.path.join(os.getcwd(), "test")

Test_Data=[]
for file in os.listdir(Test):
    if file.endswith('.jpg'):
        img_path = os.path.join(Test, file)
        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_array = np.array(img)/255.0
        Test_Data.append(img_array)
Test_Images = np.array(Test_Data)

IMAGE_SIZE = 128
NUM_CLASSES = 33
BATCH_SIZE = 32

imagegenerator = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2, rotation_range=10, horizontal_flip = True)

Train_Data = imagegenerator.flow_from_directory(
    Train,
    target_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'training'
)
Validation_Data = imagegenerator.flow_from_directory(
    Train,
    target_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'validation'
)

precision_metric = Precision()
recall_metric = Recall()

def f1_score(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

print(Train_Data.class_indices)

model = Sequential([
    Input(shape = (IMAGE_SIZE,IMAGE_SIZE,3)),
    Conv2D(32, (3,3), padding = 'same', activation = 'relu'),
    MaxPool2D(pool_size = (2,2)),
    
    Conv2D(64, (3,3), padding = 'same', activation='relu'),
    MaxPool2D(pool_size = (2,2)),
    
    Conv2D(128, (3,3), padding = 'same', activation='relu'),
    MaxPool2D(pool_size = (2,2)),

    Flatten(),
    Dense(512, activation = 'relu'),
    Dense(NUM_CLASSES, activation = 'softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), f1_score])
print(model.summary())

History = model.fit(Train_Data, validation_data=Validation_Data, epochs=2)

Test_Images[0].shape

model.evaluate(Validation_Data)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    t=random.randint(1,Test_Images.shape[0]-1)
    EachImage = np.expand_dims(Test_Images[t], axis=0)
    prediction = model.predict(EachImage)
    predicted_label = [key for key,value in Train_Data.class_indices.items() if value == np.argmax(prediction, axis=1)[0]]
    
    plt.title(predicted_label)
    plt.imshow(Test_Images[t])
    plt.axis('off')



