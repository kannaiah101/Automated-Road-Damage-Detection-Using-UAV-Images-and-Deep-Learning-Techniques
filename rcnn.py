import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import xml.etree.ElementTree as ET
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.applications import MobileNetV2 #importing InceptionResNetV2 model
from keras.applications import DenseNet201

def yolov5():
    X = np.load('model/X1.npy')
    Y = np.load('model/Y1.npy')
    bb = np.load('model/Z1.npy')
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    bb = bb[indices]
    Y = to_categorical(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)#split dataset into train and test
    input_img = Input(shape=(128, 128, 3))
    #create YoloV4 layers with 32, 64 and 512 neurons or data filteration size
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_img)
    x = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    #define output layer with 4 bounding box coordinate and 1 weapan class
    x = Dense(512, activation = 'relu')(x)
    x = Dense(512, activation = 'relu')(x)
    x_bb = Dense(12, name='bb')(x)
    x_class = Dense(y_train.shape[1], activation='softmax', name='class')(x)
    #create yolo Model with above input details
    yolo_model = Model([input_img], [x_bb, x_class])
    #compile the model
    yolo_model.compile(Adam(lr=0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'])
    if os.path.exists("model/v5.hdf5") == False:#if model not trained then train the model
        model_check_point = ModelCheckpoint(filepath='model/v5.hdf5', verbose = 1, save_best_only = True)
        hist = yolo_model.fit(X, [bb, Y], batch_size=32, epochs=20, validation_split=0.2, callbacks=[model_check_point])
        f = open('model/v5_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:#if model already trained then load it
        yolo_model.load_weights("model/v5.hdf5")
    predict = yolo_model.predict(X_test)#perform prediction on test data
    predict = np.argmax(predict[1], axis=1)
    test = np.argmax(y_test, axis=1)
    

def train():
    data = np.load('model/X1.npy')
    labels = np.load('model/Y1.npy')
    bboxes = np.load('model/Z1.npy')
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    bboxes = bboxes[indices]
    labels = to_categorical(labels)
    split = train_test_split(data, labels, bboxes, test_size=0.20, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]
    if os.path.exists("model/v7.hdf5") == False:
        rcnn = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(data.shape[1], data.shape[2], data.shape[3])))
        rcnn.trainable = False
        flatten = rcnn.output
        flatten = Flatten()(flatten)
        bboxHead = Dense(16, activation="relu")(flatten)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(12, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(16, activation="relu")(flatten)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(8, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(labels.shape[1], activation="softmax", name="class_label")(softmaxHead)
        rcnn_model = Model(inputs=rcnn.input, outputs=(bboxHead, softmaxHead))
        losses = {"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        opt = Adam(lr=1e-4)
        rcnn_model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
        trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
        testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
        model_check_point = ModelCheckpoint(filepath='model/v7.hdf5', verbose = 1, save_best_only = True)
        hist = rcnn_model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets), batch_size=32, epochs=20, verbose=1,callbacks=[model_check_point])
        f = open('model/v7.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    else:
        rcnn_model = load_model('model/v7.hdf5')
    predict = rcnn_model.predict(testImages)[1]#perform prediction on test data
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(testLabels, axis=1)
    acc = accuracy_score(testY, predict)
    print(acc)

   
yolov5()
