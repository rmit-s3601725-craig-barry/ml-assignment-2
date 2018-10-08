import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

TRAIN_SIZE = 60000
#Number of samples used per gradient increment
BATCH_SIZE = 600
#Keras docs recommend steps per epoch = number of samples / batch size
STEPS_TRAIN = TRAIN_SIZE * 0.9 / BATCH_SIZE
STEPS_VALID = TRAIN_SIZE * 0.1 / BATCH_SIZE
#Number of training epochs
EPOCHS = 1

#Structure of code taken from Keras issue #1711 - 
#https://github.com/keras-team/keras/issues/1711#issuecomment-185801662 
#Comment by KeironO
def main(args):
    train_generator, test_generator = load_data(False)
    #Read the data from the generator so it can be split
    data = []
    labels = []
    i = 0
    for d, l in train_generator:
        data.append(d)
        labels.append(l.argmax(axis=1))
        i += 1
        if i == TRAIN_SIZE / BATCH_SIZE:
            break
    
    data = np.array(data)
    #Reshape array so the number of indices is same as labels
    data = np.reshape(data, (data.shape[0]*data.shape[1],) + data.shape[2:])

    labels = np.array(labels)
    #Reduce 3d array to a 2d one, otherwise Stratified kfold won't work
    labels = np.reshape(labels, (labels.shape[0]*labels.shape[1],) + labels.shape[2:])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for i, (train_indices, val_indices) in enumerate(skf.split(data, labels)):
        print("Training on fold ", i+1, "/5...")

        #Generate batches from indices
        xtrain, xval = data[train_indices], data[val_indices]
        ytrain, yval = labels[train_indices], labels[val_indices]

        #Clear model, and create it
        model = None
        model = create_model()

        history = model.fit(
                x=xtrain,
                y=ytrain,
                steps_per_epoch=STEPS_TRAIN,
                epochs=EPOCHS
        )
        accuracy = history.history['acc']
    #Regular train/test method
    #train_generator, test_generator, valid_generator = load_data()
    #model = create_model()
    #train_and_validate(model, train_generator, test_generator, valid_generator)

    #Reset seems to be required to align estimated y to actual y
    #test_generator.reset()
    #Predict the testing data y values (classify clothing items)
    #pred = model.predict_generator(
    #        generator=test_generator,
    #        verbose=1,
    #        steps=1000
    #);
    #plot(history);

   # save(train_generator, test_generator, pred)

# load data with this function, it can split on validation as option
def load_data(validation=True):
    #Used to scale pixel values between 0 & 1
    scaleFactor = 1./255
    #Create image generator
    datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            # set aside 10% of training data for validation
            validation_split=0.2 if validation else 0
    )

    #Load training images
    train_generator = datagen.flow_from_directory(
            directory=r"./data/mnist/training/",
            target_size=(28,28),
            color_mode="grayscale",
            class_mode="categorical",
            subset="training",
            shuffle=True
    );
    
    #Load test images
    test_generator = datagen.flow_from_directory(
            directory=r"./data/mnist/testing/",
            target_size=(28,28),
            color_mode="grayscale",
            class_mode="categorical",
            #Total number of testing samples must be divisible by batch size
            batch_size=10
    );

    if validation:
        #Load validation images, they are not seen while training
        valid_generator = datagen.flow_from_directory(
                directory=r"./data/mnist/training/",
                target_size=(28,28),
                color_mode="grayscale",
                class_mode="categorical",
                subset="validation",
                shuffle=True
        );
        return train_generator, test_generator, valid_generator

    return train_generator, test_generator

# Creation of the model without data yet
def create_model():
    #The number of hidden nodes in the hidden layer, this is current optimiser
    hidden_nodes = 74

    #Number of output classes
    num_classes = 10

    #Creates a sequential neural net model
    model = Sequential()

    #Add the hidden layer
    model.add(Dense(hidden_nodes, input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten());
    #Output nodes
    model.add(Dense(num_classes, activation='sigmoid'))

    #Compile to model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def train_and_evaluate(model, train_generator, test_generator):
    #Train the model
    history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=STEPS_TRAIN,
            epochs=EPOCHS
    )

def train_and_validate(model, train_generator, test_generator, valid_generator):
    #Train the model
    history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=STEPS_TRAIN,
            validation_data=valid_generator,
            validation_steps=STEPS_VALID,
            epochs=EPOCHS
    )



#Plot the history for the sake of visualising the models performance
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
