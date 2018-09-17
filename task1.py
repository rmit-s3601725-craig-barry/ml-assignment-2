import pandas as pd;
import numpy as np;
import sklearn as sk;

from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn import cluster
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split, cross_val_score;

import graphviz

DATA_FILE = 'data/property_prices.csv'
USELESS_COLS = ['id', 'lattitude', 'longtitude', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'

#Converts all string classes to integer values
def encodeClassifiers(data, cols):
	for col in cols:
		le = preprocessing.LabelEncoder()
		le.fit(data[[col]]);
		data[[col]] = le.transform(data[[col]]);

	return data;

#Cleans up NaN values in dataset
def cleanDataset(data):
	for col in list(data):
		im = preprocessing.Imputer(strategy='mean')
		im.fit(data[[col]]);
		data[[col]] = im.transform(data[[col]]);

	return data;

#Selects features based on the kbest selector score
def select_features(feats, output):

  selector = SelectKBest(chi2, k=5)
  selector.fit(feats, output);
  mask = selector.get_support(indices=True)
  new_features = []
  feature_names = list(feats.columns.values)

  for bool, feature in zip(mask, feature_names):
      if bool:
          new_features.append(feature)

  features_to_remove = list(set(feature_names) - set(new_features));
  feats = feats.drop(features_to_remove, axis=1);

  return feats;

def main(args):
	#Load data table
	data = pd.read_table('data/property_prices.csv', delimiter=',');

	#Remove useless columns
	data = data.drop(USELESS_COLS, axis=1)
	#Remove rows where target is not known (they aren't very helpful for learning)
	data = data.loc[data[OUTPUT_COL] != 'Unknown'];

	#cleanup dataset for processing
	data = encodeClassifiers(data, CLASSIFICATION_COLS);
	data = cleanDataset(data);

	#Split data into features/target
	dataAttr = data.drop([OUTPUT_COL], axis=1);
	dataOut = data[[OUTPUT_COL]];

	dataAttr = select_features(dataAttr, dataOut);

	#Split training and testing sets
	trainFeats, testFeats, trainOutputs, testOutputs = \
    	train_test_split(dataAttr, dataOut, test_size=0.2);

   	#Create model using decision tree
	clf = tree.DecisionTreeClassifier(criterion='entropy');
	clf = clf.fit(trainFeats, trainOutputs);

	ada = ensemble.AdaBoostClassifier();
	ada = ada.fit(trainFeats, trainOutputs);

	rf = ensemble.RandomForestClassifier(criterion='entropy');
	rf = rf.fit(trainFeats, trainOutputs);

	#Perform K-Fold cross validation and get accuracy score
	m = cross_val_score(clf, dataAttr, dataOut, cv=5, scoring='accuracy')
	print("K-Foldcross validated score: %f" %np.mean(abs(m)));

	m = cross_val_score(ada, dataAttr, dataOut, cv=5, scoring='accuracy')
	print("K-Foldcross validated score: %f" %np.mean(abs(m)));

	m = cross_val_score(rf, dataAttr, dataOut, cv=5, scoring='accuracy')
	print("K-Foldcross validated score: %f" %np.mean(abs(m)));

main(None);