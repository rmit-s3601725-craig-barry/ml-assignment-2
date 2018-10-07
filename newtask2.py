import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import StratifiedKFold

TRAIN_SIZE = 60000
#Number of samples used per gradient increment
BATCH_SIZE = 128
#Keras docs recommend steps per epoch = number of samples / batch size
STEPS_TRAIN = TRAIN_SIZE * 0.9 / BATCH_SIZE
STEPS_VALID = TRAIN_SIZE * 0.1 / BATCH_SIZE
#Number of training epochs
EPOCHS = 10;

def main(args):
    #Used to scale pixel values between 0 & 1
    scaleFactor = 1./255
    #Create image generator
    datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            # set aside 10% of training data for validation
            validation_split=0.1
    )

    #The number of hidden nodes in the hidden layer, this is current optimiser
    hidden_nodes = 200

    #Number of output classes
    num_classes = 10

    #Load training images
    train_generator = datagen.flow_from_directory(
            directory=r"./data/mnist/training/",
            target_size=(28,28),
            color_mode="grayscale",
            class_mode="categorical",
            subset="training",
            shuffle=True
    );
    
    #Load validation images, they are not seen while training
    valid_generator = datagen.flow_from_directory(
            directory=r"./data/mnist/training/",
            target_size=(28,28),
            color_mode="grayscale",
            class_mode="categorical",
            subset="validation",
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

    #Creates a sequential neural net model
    model = Sequential()

    #Add the hidden layer
    model.add(Dense(hidden_nodes, input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten());
    #Output nodes
    model.add(Dense(num_classes, activation='sigmoid'))

    #Compile to model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Train the model
    history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=STEPS_TRAIN,
            validation_data=valid_generator,
            validation_steps=STEPS_VALID,
            epochs=EPOCHS
    );
    #Reset seems to be required to align estimated y to actual y
    test_generator.reset()
    #Predict the testing data y values (classify clothing items)
    pred = model.predict_generator(
            generator=test_generator,
            verbose=1,
            steps=1000
    );
    plot(history);

   # save(train_generator, test_generator, pred)

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
