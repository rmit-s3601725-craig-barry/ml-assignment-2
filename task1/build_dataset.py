def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd;
import numpy as np;
import sklearn as sk;
import operator;
import sys;

from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn import cluster
from sklearn import multiclass
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.model_selection import train_test_split, cross_val_score;
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = '../data/property_prices.csv';
USELESS_COLS = ['id', 'lattitude', 'longtitude', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'

TRAIN_FILE = 'train_data.csv'
TEST_FILE  = 'test_data.csv'

N_FEATURES = 13;


#Converts all string classes to integer values
def encodeClassifiers(data, cols):
	for col in cols:
		le = preprocessing.LabelEncoder()
		le.fit(data[[col]]);
		data[[col]] = le.transform(data[[col]]);

	return data;

#Cleans up NaN values in dataset
def cleanDataset(data):
	for i in range(0,7):
		# datasub = data.loc[data[OUTPUT_COL] == i];
		for col in list(data):
			im = preprocessing.Imputer(strategy='median')
			im.fit(data.loc[data[OUTPUT_COL] == i, [col]]);
			data.loc[data[OUTPUT_COL] == i, [col]] = im.transform(data.loc[data[OUTPUT_COL] == i, [col]]);

		# data.loc[data[OUTPUT_COL] == i] = datasub;

	return data;


#Selects features based on the kbest selector score
def select_features(feats, output, n_features):

  selector = SelectKBest(chi2, k=n_features)
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

def split_by_targets(nOutputs):
	data = [];
	for i in range(nOutputs):
		data.append(pd.read_table(DATA_FILE, delimiter=',', index_col = False));
		data[i] = data[i].loc[data[i][OUTPUT_COL] != 'Unknown'];
		data[i] = data[i].drop(USELESS_COLS, axis=1)
		data[i] = encodeClassifiers(data[i], CLASSIFICATION_COLS);
		data[i] = cleanDataset(data[i]);
		replaceVals = [x for x in range(1,nOutputs+2) if x != (i+1)];
		# print(replaceVals);
		data[i][[OUTPUT_COL]] = data[i][[OUTPUT_COL]].replace(replaceVals, 0);
		data[i][[OUTPUT_COL]] = data[i][[OUTPUT_COL]].replace([(i+1)], 1);

	return data;

def main(args):
	nOutputs = 7;

	if(len(sys.argv) > 1):
		num_features = int(sys.argv[1]);
	else:
		num_features = N_FEATURES;

	#Get the first dataset
	data = pd.read_table(DATA_FILE, delimiter=',', index_col = False);
	data = data.loc[data[OUTPUT_COL] != 'Unknown'];
	#Remove useless columns
	data = data.drop(USELESS_COLS, axis=1)
	#cleanup dataset for processing
	data = encodeClassifiers(data, CLASSIFICATION_COLS);
	data = cleanDataset(data);

	#Split data into features/target
	dataAttr = data.drop([OUTPUT_COL], axis=1);
	dataOutput = data[[OUTPUT_COL]];

	dataAttr = select_features(dataAttr, dataOutput, num_features);

	#Split to training & testing sets
	trainAttr, testAttr, trainOut, testOut = train_test_split(dataAttr, dataOutput, test_size=0.2);

	# trainAttr[OUTPUT_COL] = trainOut;
	# testAttr[OUTPUT_COL] = testOut;

	trainData = pd.concat([trainAttr, trainOut], axis=1, sort=False);
	testData  = pd.concat([testAttr,  testOut],  axis=1, sort=False);

	trainData.to_csv(TRAIN_FILE, sep=',', index=False);
	testData.to_csv(TEST_FILE, sep=',', index=False);


main(None);