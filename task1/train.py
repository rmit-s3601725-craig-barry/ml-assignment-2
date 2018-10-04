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
from sklearn import multiclass
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.model_selection import train_test_split, cross_val_score;
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

# import graphviz

DATA_FILE = 'data/property_prices.csv'
USELESS_COLS = ['id', 'address', 'date'];
CLASSIFICATION_COLS = ['price_bands', 'council_area', 'region_name', 'suburb', 'realestate_agent', 'type', 'method'];
OUTPUT_COL = 'price_bands'
DEBUG = True;

TRAIN_FILE = 'train_data.csv'
TEST_FILE  = 'test_data.csv'

def get_accuracy(classifiers, feats, targets):
	probs = [];
	for clf in classifiers:
		probs.append(clf.predict_proba(feats)[:,1]);

	predictedOutputs = [];
	for i in range(len(probs[0])):
		probabilities = [probs[x][i] for x in range(len(probs))];
		max_idx, max_val = max(enumerate(probabilities), key=operator.itemgetter(1))
		predictedOutputs.append(max_idx + 1);

	accuracy = metrics.accuracy_score(targets, predictedOutputs);
	return accuracy;

def split_by_targets(nOutputs):
	data = [];
	d = pd.read_table(TRAIN_FILE, delimiter=',')
	
	for i in range(nOutputs):
		tbl = pd.read_table(TRAIN_FILE, delimiter=',')
		data.append(tbl);
		# data[i] = data[i].loc[data[i][OUTPUT_COL] != 'Unknown'];
		# data[i] = data[i].drop(USELESS_COLS, axis=1)
		# data[i] = encodeClassifiers(data[i], CLASSIFICATION_COLS);
		# data[i] = cleanDataset(data[i]);
		replaceVals = [x for x in range(1,nOutputs+2) if x != (i+1)];
		# print(replaceVals);
		data[i][[OUTPUT_COL]] = data[i][[OUTPUT_COL]].replace(replaceVals, 0);
		data[i][[OUTPUT_COL]] = data[i][[OUTPUT_COL]].replace([(i+1)], 1);

	out = d[[OUTPUT_COL]];
	tbl = d.drop(OUTPUT_COL, axis=1);

	return data, tbl, out;

def pr(msg):
	if(DEBUG):
		print(msg);


def main(args):
	clfs = [];
	adas = [];
	rfs = [];
	baggings = [];
	etcs = [];
	nOutputs = 6;

	datas, dataAttr, dataOutput = split_by_targets(nOutputs);

	#Get the first dataset
	testData = pd.read_table(TEST_FILE, delimiter=',');
	# print testData;
	# data = data.loc[data[OUTPUT_COL] != 'Unknown'];
	# #Remove useless columns
	# data = data.drop(USELESS_COLS, axis=1)
	# #cleanup dataset for processing
	# data = encodeClassifiers(data, CLASSIFICATION_COLS);
	# data = cleanDataset(data);
	# #Split data into features/target
	testDataAttr = testData.drop([OUTPUT_COL], axis=1);
	testDataOutput = testData[[OUTPUT_COL]].values.flatten();
	j = [];

	ovr = multiclass.OneVsOneClassifier(tree.DecisionTreeClassifier(criterion='entropy', max_depth=50));
	ovr = ovr.fit(dataAttr, dataOutput);
	m = cross_val_score(ovr, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	pr("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));
	j.append(np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(ensemble.AdaBoostClassifier());
	ovr = ovr.fit(dataAttr, dataOutput);
	m = cross_val_score(ovr, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	pr("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));
	j.append(np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(ensemble.RandomForestClassifier(criterion='entropy'));
	ovr = ovr.fit(dataAttr, dataOutput);
	m = cross_val_score(ovr, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	pr("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));
	j.append(np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(BaggingClassifier(KNeighborsClassifier(),max_samples=1.0, max_features=0.9));
	ovr = ovr.fit(dataAttr, dataOutput);
	m = cross_val_score(ovr, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	pr("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));
	j.append(np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10, min_samples_split=2));
	ovr = ovr.fit(dataAttr, dataOutput);
	m = cross_val_score(ovr, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	pr("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));
	j.append(np.mean(abs(m)));


	# for data in datas:
	# 	#Get the data column where only 1 classifier is present
	# 	dataOut = data[[OUTPUT_COL]].values.flatten();

	# 	#Create model using decision tree
	# 	clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=50);
	# 	clf = clf.fit(dataAttr, dataOut);

	# 	ada = ensemble.AdaBoostClassifier();
	# 	ada = ada.fit(dataAttr, dataOut);

	# 	rf = ensemble.RandomForestClassifier(criterion='entropy');
	# 	rf = rf.fit(dataAttr, dataOut);

	# 	bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=1.0, max_features=0.9);
	# 	bagging = bagging.fit(dataAttr, dataOut);

	# 	etc = ExtraTreesClassifier(n_estimators=10, min_samples_split=2);
	# 	etc = etc.fit(dataAttr, dataOut);

	# 	clfs.append(clf);
	# 	adas.append(ada);
	# 	rfs.append(rf);
	# 	baggings.append(bagging);
	# 	etcs.append(etc);

	# 	pr("PREDICTING CLASSIFIER %d" %(len(clfs)));

	# 	#Perform K-Fold cross validation and get accuracy score
	# 	m = cross_val_score(clf, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	# 	pr("Decision Tree Classifier validated score: %f" %np.mean(abs(m)));

	# 	m = cross_val_score(ada, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	# 	pr("AdaBoost Classifier validated score: %f" %np.mean(abs(m)));

	# 	m = cross_val_score(rf, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	# 	pr("Random Forest Classifier validated score: %f" %np.mean(abs(m)));

	# 	m = cross_val_score(bagging, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	# 	pr("Bagging Classifier validated score: %f" %np.mean(abs(m)));

	# 	m = cross_val_score(etc, testDataAttr, testDataOutput, cv=5, scoring='accuracy')
	# 	pr("Extra Trees Classifier validated score: %f" %np.mean(abs(m)));

	# pr("--- FINAL -> Predicting all classifiers ---");
	# mean_dtc = get_accuracy(clfs, testDataAttr, testDataOutput);
	# pr("Decision Tree Classifier: %f" %mean_dtc);
	# mean_ada = get_accuracy(adas, testDataAttr, testDataOutput);
	# pr("AdaBoost Classifier: %f" %mean_ada);
	# mean_rfc = get_accuracy(rfs, testDataAttr, testDataOutput)
	# pr("Random Forest Classifier: %f" %mean_rfc);
	# mean_bag = get_accuracy(baggings, testDataAttr, testDataOutput);
	# pr("Bagging Tree Classifier: %f" %mean_bag);
	# mean_etc = get_accuracy(etcs, testDataAttr, testDataOutput)
	# pr("Extra Trees Classifier: %f" %mean_etc);

	# print('%f,%f,%f,%f,%f' %(mean_dtc, mean_ada, mean_rfc, mean_bag, mean_etc));

	# return [mean_dtc, mean_ada, mean_rfc, mean_bag, mean_etc];

main(None);