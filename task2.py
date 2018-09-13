from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten

def main(args):
	datagen = ImageDataGenerator()

	train_generator = datagen.flow_from_directory(
		directory=r"./data/mnist/training/",
		target_size=(28,28),
		color_mode="grayscale",
		class_mode="categorical",
		shuffle=True,
		seed=42
	);

	test_generator = datagen.flow_from_directory(
		directory=r"./data/mnist/testing/",
		target_size=(28,28),
		color_mode="grayscale",
		class_mode="categorical"
	);

	model = Sequential()
	model.add(Dense(28*28, input_shape=(28, 28, 1), activation='relu'))
	model.add(Flatten());
	model.add(Dense(10, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit_generator(train_generator)

	# for i in x:
	# 	print(i);

	# print x;

main(None);