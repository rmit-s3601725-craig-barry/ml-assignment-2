def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd;
import numpy as np;
import sklearn as sk;
import operator;

from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn import cluster
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, Imputer, LabelEncoder


DATA_FILE = 'data/property_prices.csv';
CLEANED_DATA_FILE = "cleaned_data_file.csv"
USELESS_COLS = ['id', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'
N_OUTPUTS = 7;
DEBUG = True;

def pr(msg):
	if(DEBUG):
		print(msg);

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
		for col in list(data):
			im = Imputer(strategy='median')
			im.fit(data.loc[data[OUTPUT_COL] == i, [col]]);
			data.loc[data[OUTPUT_COL] == i, [col]] = im.transform(data.loc[data[OUTPUT_COL] == i, [col]]);

	return data;

#Selects features based on the kbest selector score
def select_features(feats, output, n_features):
	selector = SelectKBest(f_classif, k=n_features)
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

def build_dataset(n_features):
	print('--Building Dataset--');

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

	dataAttr = select_features(dataAttr, dataOutput, n_features);

	data = pd.concat([dataAttr, dataOutput], axis=1, sort=False);

	data.to_csv(CLEANED_DATA_FILE, sep=',', index=False);

def process(clfScores, desc, clf, dataAttr, dataOutput, kFolds):
	clfs = [
		(desc,clf),
		("%s (One-vs-One)" %desc,OneVsOneClassifier(clf)),
		("%s (One-vs-Rest)" %desc,OneVsRestClassifier(clf)),
		("AdaBoost %s" %desc,AdaBoostClassifier(clf)),
		("AdaBoost %s (One-vs-One)" %desc,OneVsOneClassifier(AdaBoostClassifier(clf))),
		("AdaBoost %s (One-vs-Rest)" %desc,OneVsRestClassifier(AdaBoostClassifier(clf))),
		("Bagging %s" %desc,BaggingClassifier(clf)),
		("Bagging %s (One-vs-One)" %desc,OneVsOneClassifier(BaggingClassifier(clf))),
		("Bagging %s (One-vs-Rest)" %desc,OneVsRestClassifier(BaggingClassifier(clf))),
	];

	for i in range(len(clfs)):
		description = clfs[i][0];
		clf = clfs[i][1];

		cvs = cross_val_score(clf, dataAttr, dataOutput, cv=kFolds, scoring='accuracy');
		accuracy_score = np.mean(abs(cvs)) * 100.0;
		pr("%s: %.2f%%" %(description, accuracy_score));
		clfScores.append((description, accuracy_score));

	pr('');

def train(kfolds):
	clfScores = [];
	num_top_classifiers = 10;

	#Retrieve data
	data = pd.read_table(CLEANED_DATA_FILE, delimiter=',');
	dataOutput = data[[OUTPUT_COL]];
	dataAttr = data.drop(OUTPUT_COL, axis=1);

	################################
	# classifiers
	################################
	pr('\n-- Classifiers --');

	clf = DecisionTreeClassifier(criterion='entropy');
	process(clfScores, "Decision Tree Entropy Classifier", clf, dataAttr, dataOutput, kfolds);

	clf = DecisionTreeClassifier(criterion='gini');
	process(clfScores, "Decision Tree Gini Classifier", clf, dataAttr, dataOutput, kfolds);

	clf = ExtraTreesClassifier(criterion='entropy', n_estimators=10, min_samples_split=2);
	process(clfScores, "Extra Trees Entropy Classifier", clf, dataAttr, dataOutput, kfolds);

	clf = ExtraTreesClassifier(criterion='gini', n_estimators=10, min_samples_split=2);
	process(clfScores, "Extra Trees Gini Classifier", clf, dataAttr, dataOutput, kfolds);

	clf = RandomForestClassifier(criterion='entropy');
	process(clfScores, "Random Forest Entropy Classifier", clf, dataAttr, dataOutput, kfolds);

	clf = RandomForestClassifier(criterion='gini');
	process(clfScores, "Random Forest Gini Classifier", clf, dataAttr, dataOutput, kfolds);

	################################
	# Top Classifiers
	################################
	pr('\n-- Top %d Decision Tree Classifiers --' %num_top_classifiers);

	#Sort classifiers by accuracy score descending 
	sortedClassifierList = sorted(clfScores, key=lambda x: -x[1]);
	#Print the top classifiers
	for i in range(num_top_classifiers):
		pr("%d. %s: %.2f%%" %((i+1), sortedClassifierList[i][0], sortedClassifierList[i][1]));

def main(args):
	n_features = 12;
	kfolds = 3;

	build_dataset(n_features);
	train(kfolds);

main(None);