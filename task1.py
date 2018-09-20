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
	for i in range(0,7):
		# datasub = data.loc[data[OUTPUT_COL] == i];
		for col in list(data):
			im = preprocessing.Imputer(strategy='median')
			im.fit(data.loc[data[OUTPUT_COL] == i, [col]]);
			data.loc[data[OUTPUT_COL] == i, [col]] = im.transform(data.loc[data[OUTPUT_COL] == i, [col]]);

		# data.loc[data[OUTPUT_COL] == i] = datasub;

	return data;

def cleanDataset2(data):

	for col in list(data):
		im = preprocessing.Imputer(strategy='median')
		im.fit(data[[col]]);
		data[[col]] = im.transform(data[[col]]);

		# data.loc[data[OUTPUT_COL] == i] = datasub;



	return data;


#Selects features based on the kbest selector score
def select_features(feats, output):

  selector = SelectKBest(f_regression, k=5)
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

def split_by_targets(nOutputs, splits = 2):
	data = [];
	for i in range(nOutputs):
		data.append(pd.read_table('data/property_prices.csv', delimiter=','));
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
	clfs = [];
	adas = [];
	rfs = [];
	baggings = [];
	etcs = [];
	nOutputs = 6;

	datas = split_by_targets(nOutputs);

	#Get the first dataset
	data = pd.read_table('data/property_prices.csv', delimiter=',');
	data = data.loc[data[OUTPUT_COL] != 'Unknown'];
	#Remove useless columns
	data = data.drop(USELESS_COLS, axis=1)
	#cleanup dataset for processing
	data = encodeClassifiers(data, CLASSIFICATION_COLS);
	# print data;
	# data = data.dropna();
	# print(data);
	data = cleanDataset(data);
	# print(data);
	# print data;
	#Split data into features/target
	dataAttr = data.drop([OUTPUT_COL], axis=1);
	dataOutput = data[[OUTPUT_COL]];
	# dataAttr = select_features(dataAttr, dataOutput);
	#Split training and testing sets
	# trainFeats, testFeats, trainOutputs, testOutputs = \
 #    	train_test_split(dataAttr, dataOut, test_size=0.2);

 	ovr = multiclass.OneVsOneClassifier(tree.DecisionTreeClassifier(criterion='entropy', max_depth=50));
 	ovr = ovr.fit(dataAttr, dataOutput);
 	m = cross_val_score(ovr, dataAttr, dataOutput, cv=5, scoring='accuracy')
	print("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(ensemble.AdaBoostClassifier());
 	ovr = ovr.fit(dataAttr, dataOutput);
 	m = cross_val_score(ovr, dataAttr, dataOutput, cv=5, scoring='accuracy')
	print("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(ensemble.RandomForestClassifier(criterion='entropy'));
 	ovr = ovr.fit(dataAttr, dataOutput);
 	m = cross_val_score(ovr, dataAttr, dataOutput, cv=5, scoring='accuracy')
	print("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(BaggingClassifier(KNeighborsClassifier(),max_samples=1.0, max_features=0.9));
 	ovr = ovr.fit(dataAttr, dataOutput);
 	m = cross_val_score(ovr, dataAttr, dataOutput, cv=5, scoring='accuracy')
	print("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));

	ovr = multiclass.OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10, min_samples_split=2));
 	ovr = ovr.fit(dataAttr, dataOutput);
 	m = cross_val_score(ovr, dataAttr, dataOutput, cv=5, scoring='accuracy')
	print("One Vs Rest Classifier validated score: %f" %np.mean(abs(m)));


	for data in datas:
		#Get the data column where only 1 classifier is present
		dataOut = data[[OUTPUT_COL]];

	   	#Create model using decision tree
		clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=50);
		clf = clf.fit(dataAttr, dataOut);

		ada = ensemble.AdaBoostClassifier();
		ada = ada.fit(dataAttr, dataOut);

		rf = ensemble.RandomForestClassifier(criterion='entropy');
		rf = rf.fit(dataAttr, dataOut);

		bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=1.0, max_features=0.9);
		bagging = bagging.fit(dataAttr, dataOut);

		etc = ExtraTreesClassifier(n_estimators=10, min_samples_split=2);
		etc = etc.fit(dataAttr, dataOut);

		clfs.append(clf);
		adas.append(ada);
		rfs.append(rf);
		baggings.append(bagging);
		etcs.append(etc);

		print("PREDICTING CLASSIFIER %d" %(len(clfs)));

		#Perform K-Fold cross validation and get accuracy score
		m = cross_val_score(clf, dataAttr, dataOut, cv=5, scoring='accuracy')
		print("Decision Tree Classifier validated score: %f" %np.mean(abs(m)));

		m = cross_val_score(ada, dataAttr, dataOut, cv=5, scoring='accuracy')
		print("AdaBoost Classifier validated score: %f" %np.mean(abs(m)));

		m = cross_val_score(rf, dataAttr, dataOut, cv=5, scoring='accuracy')
		print("Random Forest Classifier validated score: %f" %np.mean(abs(m)));

		m = cross_val_score(bagging, dataAttr, dataOut, cv=5, scoring='accuracy')
		print("Bagging Classifier validated score: %f" %np.mean(abs(m)));

		m = cross_val_score(etc, dataAttr, dataOut, cv=5, scoring='accuracy')
		print("Extra Trees Classifier validated score: %f" %np.mean(abs(m)));

	print("--- FINAL -> Predicting all classifiers ---");
	print("Decision Tree Classifier: %f" %get_accuracy(clfs, dataAttr, dataOutput));
	print("AdaBoost Classifier: %f" %get_accuracy(adas, dataAttr, dataOutput));
	print("Random Forest Classifier: %f" %get_accuracy(rfs, dataAttr, dataOutput));
	print("Bagging Tree Classifier: %f" %get_accuracy(baggings, dataAttr, dataOutput));
	print("Extra Trees Classifier: %f" %get_accuracy(etcs, dataAttr, dataOutput));








	# maxScore = 0;
	# maxDX = 0;
	# maxDY = 0;
	# for x in range(11):
	# 	for y in range(11):
	# 		dx = x * 0.1;
	# 		dy = y * 0.1;

	# 		try:
	# 			etc = ExtraTreesClassifier(n_estimators=10, min_samples_split=2);
	# 			etc = etc.fit(trainFeats, trainOutputs);

	# 			m = np.mean(abs(cross_val_score(etc, dataAttr, dataOut, cv=5, scoring='accuracy')));
	# 			print("(%f,%f) score: %f" %(dx, dy, m));

	# 			if(m > maxScore):
	# 				maxDX = dx;
	# 				maxDY = dy;
	# 				maxScore = m;
	# 		except:
	# 			pass;

	# print("Final: (%f, %f), score: %f" %(maxDX, maxDY, maxScore));

main(None);