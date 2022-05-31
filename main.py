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

from tensorflow import keras
from mySVM import SupportVectorMachine
          



def extractFeature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    mfcc2 = kwargs.get("mfcc2")

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        # print("sample rate = ", sample_rate)
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if mfcc2:
            from mfccFinal import MFCC
            mfccs2 = MFCC(sample=X, sampleRate= sample_rate, filtersAmount= 40) 
            mfccs2 = np.mean(mfccs2.mfcc().T, axis = 0)
            result = np.hstack((result, mfccs2))
    return result


def dataLoader(emotionType, test_size=0.01):
    X, y = [], []
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

    AVAILABLE_EMOTIONS = {
        "angry": 147,
        "sad": 131,
        "happy": 151
    }
    
    emotionCounter = 0
    otherCounter = 0
    for file in glob.glob("data/Actor_*/*.wav"):

        basename = os.path.basename(file)
        emotion = int2emotion[basename.split("-")[2]]

        if emotion not in AVAILABLE_EMOTIONS:
            continue

        if emotionCounter < AVAILABLE_EMOTIONS[emotionType] and emotion == emotionType:
            features = extractFeature(file, mfcc2=True)
            X.append(features)
            y.append(1.0)
            emotionCounter += 1
        if otherCounter < AVAILABLE_EMOTIONS[emotionType] and emotion != emotionType:
            features = extractFeature(file, mfcc2=True)
            X.append(features)
            y.append(-1.0)
            otherCounter += 1
            

    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

X_train, X_test, y_train, y_test = dataLoader("happy", test_size=0.01)
y_train_new = []
y_test_new = []

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train_new = np.array(y_train) 
y_test_new = np.array(y_test)

def kernel(x, y, sigma=1):
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        return np.exp(-np.power(np.linalg.norm(x-y),2)/2*sigma**2)
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
      result = []
      temp = []
      for i in range(len(x)):
        for j in range(len(y)):
          diff = x[i]-y[j]
          temp_result = np.exp(-np.power(np.linalg.norm(diff),2)/2*sigma**2)
          temp.append(temp_result)
        result.append(temp)
        temp = []
      return np.array(result)
    else:
        return np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))

if __name__ == "__main__":
    from recordVoice import record
    record()
    X_test_voice = extractFeature("input.wav", mfcc2=True)
    print(X_test_voice)
    X_test_voice = np.array([X_test_voice])
    # model = SupportVectorMachine(C = 1,kernel = kernel)
    # model.fit(X_train,y_train_new)
    # ar = model.predict(X_test_voice)
    # print("ar = ",ar)


    # accuracy = accuracy_score(y_true=y_test_new, y_pred=ar)

    # print("Accuracy: {:.2f}%".format(accuracy*100))

    # saved_model = pickle.dump(model, open("modelHappy.model", "wb"))

    
    loaded_modelHappy = pickle.load(open("modelHappy.model",'rb'))
    loaded_modelSad = pickle.load(open("modelSad.model",'rb'))
    loaded_modelAngry = pickle.load(open("modelAngry.model",'rb'))
    modelsArr = [loaded_modelHappy, loaded_modelSad, loaded_modelAngry]
    lst_emotions = []
    for i in range(3):
        lst_emotions.append(modelsArr[i].predict(X_test_voice))
    print(lst_emotions)
    
    max_index = lst_emotions.index(max(lst_emotions))
    if max_index == 0:
        print("happy")
    elif max_index == 1:
        print("sad")
    elif max_index == 2:
        print("angry")
        
