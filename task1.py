import pandas as pd;
import numpy as np;
import sklearn as sk;

from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score;

import graphviz

DATA_FILE = 'data/property_prices.csv'
USELESS_COLS = ['id', 'lattitude', 'longtitude', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'

def encodeClassifiers(data, cols):
	for col in cols:
		le = preprocessing.LabelEncoder()
		le.fit(data[[col]]);
		data[[col]] = le.transform(data[[col]]);

	return data;

def cleanDataset(data):
	
	
	for col in list(data):
		im = preprocessing.Imputer(strategy='median')
		im.fit(data[[col]]);
		data[[col]] = im.transform(data[[col]]);

	return data;


def main(args):
	data = pd.read_table('data/property_prices.csv', delimiter=',');

	data = data.drop(USELESS_COLS, axis=1)
	data = data.loc[data[OUTPUT_COL] != 'Unknown'];

	data = encodeClassifiers(data, CLASSIFICATION_COLS);
	data = cleanDataset(data);

	dataAttr = data.drop([OUTPUT_COL], axis=1);
	dataOut = data[[OUTPUT_COL]];

	le = preprocessing.LabelEncoder();
	le.fit(dataOut);
	dataOut = le.transform(dataOut);

	trainFeats, testFeats, trainOutputs, testOutputs = \
    	train_test_split(dataAttr, dataOut, test_size=0.2);

	# print(data);
	# print(dataOut);

	clf = tree.DecisionTreeClassifier(criterion='entropy');
	clf = clf.fit(trainFeats, trainOutputs);
	m = cross_val_score(clf, dataAttr, dataOut, cv=5, scoring='accuracy')
	print("K-Foldcross validated score: %f" %np.mean(abs(m)));
	
	
	# print(len(data));
	pass;

main(None);