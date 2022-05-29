import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.svm import SVC # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from scipy import optimize

import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from sklearn import linear_model, svm, discriminant_analysis, metrics
from scipy import optimize

          



def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    mfcc2 = kwargs.get("mfcc2")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
            # print(result)
        
        if mfcc2:
            from mfccFinal import MFCC
            mfccs2 = MFCC(sample=X, sampleRate= sample_rate, filtersAmount= 40) 
            mfccs2 = np.mean(mfccs2.mfcc().T, axis = 0)
            result = np.hstack((result, mfccs2))
    return result



int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions ( feel free to tune this on your need )
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

def load_data(test_size=0.2):
    X, y = [], []
    counter = 0
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # print(basename)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc2=True)
        # add to data
        X.append(features)
        y.append(emotion)
        counter += 1 
        if counter == 100:
            break
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# X_train2, X_test2, y_train2, y_test2 = load_data2(test_size=0.25)

y_train_new = [] 
y_test_new = []

for i in y_train:
    if i == "happy":
        y_train_new.append(1.0)
    else:
        y_train_new.append(-1.0)

        
for i in y_test:
    if i == "happy":
        y_test_new.append(1.0)
    else:
        y_test_new.append(-1.0)
        

X_train = np.array(X_train)
X_test = np.array(X_test)

# X_train2 = np.array(X_train2)
# X_test2 = np.array(X_test2)
y_train_new = np.array(y_train_new) 
# print(y_train_new)
y_test_new = np.array(y_test_new)


# X_train = np.array(X_train,dtype=np.float32)
# y_train = np.array(y_train,dtype=np.float32)
# X_test = np.array(X_test,dtype=np.float32)
# y_test = np.array(y_test,dtype=np.float32)

# print(len(y_train))

# # print some details
# # number of samples in training data
# print("[+] Number of training samples:", X_train.shape[0])
# # number of samples in testing data
# print("[+] Number of testing samples:", X_test.shape[0])
# # number of features used
# # this is a vector of features extracted 
# # using extract_features() function
# print("[+] Number of features:", X_train.shape[1])

from SVM import *
model = KernelSvmClassifier(C = 1, kernel = GRBF)
print("START FIT")
model.fit(X_train,y_train_new)
print("End FIT")
y_pred = model.predict(X_test)
print(y_pred)
print(y_test_new)
print("END PREDICT")

# from mySVM import SupportVectorMachine
# model2 = SupportVectorMachine(C=1, kernel="g")
# a = model2.fit(X_train, y_train_new)


accuracy = accuracy_score(y_true=y_test_new, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
