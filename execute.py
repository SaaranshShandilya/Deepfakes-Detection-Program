# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:27:05 2021

@author: saaransh
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
#its loading the contents of the file my laptop is a kinda slow so thats why its taking time
cap= cv2.VideoCapture('C:/Users/saaransh/Downloads/train_sample_videos/btiysiskpf.mp4')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%100 == 0:
        
        cv2.imwrite(os.path.join('./images' , str(i) + '.jpg'),frame)
        image = cv2.imread('./images/{}.jpg'.format(i))
            # Convert the image to RGB colorspace
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make a copy of the original image to draw face detections on
        image_copy = np.copy(image)
            
            # Convert the image to gray 
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)
            
            # Print number of faces found
        print('Number of faces detected:', len(faces))
            
            # Get the bounding box for each detected face
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
            # Define the region of interest in the image  
            face_crop = gray_image[y:y+h, x:x+w]
            
            # Display the image with the bounding boxes
        fig = plt.figure(figsize = (9,9))
        axl = fig.add_subplot(111)
        axl.set_xticks([])
        axl.set_yticks([])
        cv2.imwrite(os.path.join('./facial only' , str(i) + '.jpg'),face_crop)

    i+=1
    
cap.release()
cv2.destroyAllWindows()

image_dimensions = {'height':256, 'width':256, 'channels':3}

class Classifier:
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)
        
class Meso4(Classifier):
    def __init__(self,learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean squared error',
                           metrics = ['accuracy'])
        
    def init_model(self):
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)
    
meso = Meso4()
meso.load('C:/Users/saaransh/Downloads/Meso4_DF')

dataGenerator = ImageDataGenerator(rescale=1./255)

# Instantiating generator to feed images through the network
generator = dataGenerator.flow_from_directory(
    './',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')

# Rendering image X with label y for MesoNet
X, y = generator.next()

# Evaluating prediction
print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
print(f"Actual label: {int(y[0])}")
print(f"\nCorrect prediction: {round(meso.predict(X)[0][0])==y[0]}")

# Showing image
plt.imshow(np.squeeze(X));

#The model predicts the probability of Deepfakes, so the more closer it is to zero the more likely it is a deepfake and viceversa


