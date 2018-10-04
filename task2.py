from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten

def main(args):
	#Used to scale pixel values between 0 & 1
	scaleFactor = 1./255;
	#Create image generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)

	batch_size = 128;
	#Steps per epoch set as recommended by the keras documentation
	#as number of samples divided by batch size
	steps_per_epoch = 60000 / batch_size;

	#The number of hidden nodes in the hidden layer
	hidden_nodes = 1024

	#Number of output classes
	num_classes = 10;
	#Number of training epochs
	epochs = 10;

	#Load training images
	train_generator = datagen.flow_from_directory(
		directory=r"./data/mnist/training/",
		target_size=(28,28),
		color_mode="grayscale",
		class_mode="categorical",
		shuffle=True
	);

	#Load test images
	test_generator = datagen.flow_from_directory(
		directory=r"./data/mnist/testing/",
		target_size=(28,28),
		color_mode="grayscale",
		class_mode="categorical"
	);

	#Creates a sequential neural net model
	model = Sequential()

	#Add the hidden layer
	model.add(Dense(hidden_nodes, input_shape=(28, 28, 1), activation='relu'))
	model.add(Flatten());
	#Output nodes
	model.add(Dense(num_classes, activation='sigmoid'))

	#Compile to model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
	#Train the model
	model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs);

main(None);