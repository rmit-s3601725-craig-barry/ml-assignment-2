# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import skimage.io
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from statistics import stdev

#Partition of samples used for validation
KFOLDS = 5
#Number of training epochs
EPOCHS = 10

def main(args):
    print('Loading images...');

    #Path to load training images from
    train_data_path = os.path.join("data", "mnist", "training");
    test_data_path = os.path.join("data", "mnist", "testing");
    
    #Retrieve the training inputs & output classes
    xTrain, yTrain = load_data(train_data_path);
    xTest, yTest = load_data(test_data_path);
    
    model = train_model(xTrain, yTrain);
    scores = test_model(model, xTest, yTest);

    print("MLP Error on test set: %.2f%%" % (100-scores[1]*100));
    print("MLP Accuracy on test set: %.2f%%" % (scores[1]*100));

def train_model(xTrain, yTrain):
    #Get the number of pixels in each image
    num_pixels = xTrain.shape[1] * xTrain.shape[2];
    #Flatten pixel arrau
    xTrain = xTrain.reshape(xTrain.shape[0], num_pixels).astype('float32');
    #Get the labels for the output lasses
    labels = yTrain;
    #Convert labels to categorical training set
    yTrain = np_utils.to_categorical(yTrain);
    #Get the number of output classes
    num_classes = yTrain.shape[1];
    #Create kfold cross validation splitter
    kfold = StratifiedKFold(n_splits=KFOLDS, shuffle=True);

    print('Preprocessing...');

    crossValScores = [];

    #Split training & testing sets using kfold validation
    for i, (trainIdxs, testIdxs) in enumerate(kfold.split(xTrain, labels)):


        print('Creating Model...');

        #Create neural net model
        model = create_model(num_pixels);

        print("Training on fold ", i+1, "/ 5...")

        #Fit model to training data
        model.fit(xTrain[trainIdxs], yTrain[trainIdxs], epochs=EPOCHS);

        #Evaliate model using testing data
        scores = model.evaluate(xTrain[testIdxs], yTrain[testIdxs], verbose = 0);

        print("MLP Error on validation set: %.2f%%" % (100-scores[1]*100));
        print("MLP Accuracy on validation set: %.2f%%" % (scores[1]*100));

        crossValScores.append(scores[1]*100.0);


    print("MLP Score Standard Deviation: %.2f%%" %stdev(crossValScores))
    print("MLP Average Accuracy: %.2f%%" %(sum(crossValScores)/len(crossValScores)));
    return model

def test_model(model, xTest, yTest):
    #Get the number of pixels in each image
    num_pixels = xTest.shape[1] * xTest.shape[2];
    #Flatten pixel arrau
    xTest = xTest.reshape(xTest.shape[0], num_pixels).astype('float32');
    yTest = np_utils.to_categorical(yTest);
    #Evaluate model using testing data
    return model.evaluate(xTest, yTest, verbose = 1);


#Load images from given directory
def load_data(dataDir):
    dirs = [d for d in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, d))];

    labels = [];
    images = [];
    for d in dirs:

        labelDir = os.path.join(dataDir, d);
        fileNames = [os.path.join(labelDir, f)
            for f in os.listdir(labelDir)];

        for file in fileNames:
            images.append(skimage.io.imread(file) / 255.0);
            labels.append(int(d));

    return np.asarray(images), np.asarray(labels);

def create_model(num_pixels):
    #The number of hidden nodes in the hidden layer, this is current optimiser
    hidden_nodes = 250

    #Number of output classes
    num_classes = 10

    #Creates a sequential neural net model
    model = Sequential()

    #Add the hidden layer
    model.add(Dense(hidden_nodes, input_dim=num_pixels, activation='relu'))
    # model.add(Flatten());
    #Output nodes
    model.add(Dense(num_classes, activation='sigmoid'))

    #Compile to model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot(history):
    #print(history.history.keys())
    #Summarise history for accuracy
    #Keys come in history object
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #Summarise history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def save(train_generator, test_generator, pred):
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_csv("results.csv",index=False)

main(None);
