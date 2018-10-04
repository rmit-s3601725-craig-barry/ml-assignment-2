def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd;
import numpy as np;
import sklearn as sk;
import operator;
import sys;

from scipy import stats
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import cluster
from sklearn import multiclass
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score;
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, Imputer, LabelEncoder

DATA_FILE = '../data/property_prices.csv';
CLEANED_DATA_FILE = "cleaned_data_file.csv"
USELESS_COLS = ['id', 'lattitude', 'longtitude', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'

N_FEATURES = 13;
N_OUTPUTS = 7;

#Converts all string classes to integer values
def encodeClassifiers(data, cols):
	for col in cols:
		le = LabelEncoder()
		le.fit(data[[col]]);
		data[[col]] = le.transform(data[[col]]);

	return data;

#Cleans up NaN values in dataset
def cleanDataset(data):
	for i in range(0,N_OUTPUTS):
		# datasub = data.loc[data[OUTPUT_COL] == i];
		for col in list(data):
			im = Imputer(strategy='median')
			im.fit(data.loc[data[OUTPUT_COL] == i, [col]]);
			data.loc[data[OUTPUT_COL] == i, [col]] = im.transform(data.loc[data[OUTPUT_COL] == i, [col]]);

	return data;


#Selects features based on the kbest selector score
def select_features(feats, output, n_features):

  # selector = SelectKBest(mutual_info_classif, k=n_features)
  selector = VarianceThreshold(threshold=0.80);
  
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

	data = pd.concat([dataAttr, dataOutput], axis=1, sort=False);

	data.to_csv(CLEANED_DATA_FILE, sep=',', index=False);

main(None);