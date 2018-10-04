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
from sklearn.model_selection import train_test_split, cross_val_score;
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

CLEANED_DATA_FILE = "cleaned_data_file.csv"
USELESS_COLS = ['id', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'
DEBUG = True;

TRAIN_FILE = 'train_data.csv'
TEST_FILE  = 'test_data.csv'

def pr(msg):
	if(DEBUG):
		print(msg);

def process(clfScores, desc, clf, dataAttr, dataOutput, kFolds):
	clfs = [
		clf, 
		OneVsOneClassifier(clf), 
		OneVsRestClassifier(clf),
		AdaBoostClassifier(clf), 
		OneVsOneClassifier(AdaBoostClassifier(clf)),
		OneVsRestClassifier(AdaBoostClassifier(clf)),
		BaggingClassifier(clf),
		OneVsOneClassifier(BaggingClassifier(clf)),
		OneVsRestClassifier(BaggingClassifier(clf))
	];
	descs = [
		desc,
	 	"%s (One-vs-One)" %desc,
	 	"%s (One-vs-Rest)" %desc,
	 	"AdaBoost %s" %desc,
	 	"AdaBoost %s (One-vs-One)" %desc,
	 	"AdaBoost %s (One-vs-Rest)" %desc,
	 	"Bagging %s" %desc,
	 	"Bagging %s (One-vs-One)" %desc,
	 	"Bagging %s (One-vs-Rest)" %desc,
	 ];

	for i in range(len(clfs)):
		clf = clfs[i];
		desc = descs[i]

		cvs = cross_val_score(clf, dataAttr, dataOutput, cv=kFolds, scoring='accuracy');
		accuracy_score = np.mean(abs(cvs));
		pr("%s: %f" %(desc, accuracy_score));
		clfScores.append((desc, accuracy_score));

	pr('');

def main(args):
	clfScores = [];
	kfolds = 3;
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
	pr('\n-- Top 5 Decision Tree Classifiers --');

	#Sort classifiers by accuracy score descending 
	sortedClassifierList = sorted(j, key=lambda x: -x[1]);
	#Print the top classifiers
	for i in range(num_top_classifiers):
		pr("%d. %s: %f" %((i+1), sortedClassifierList[i][0], sortedClassifierList[i][1]));

main(None);