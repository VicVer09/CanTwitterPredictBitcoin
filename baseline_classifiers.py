# basic modules
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
import sys 
import pickle
import math
import matplotlib.pyplot as plt

# import classifiers 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier

from sklearn.feature_extraction.text import CountVectorizer 

# import hyperparameter optimization tools
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, explained_variance_score, r2_score

smoothing = [0.0, 0.00001, 0.00005, 0.0001, 0.00001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
col_names = {		0: '1 Min Delta',
					1: '5 Min Delta',
					2: '60 Min Delta',
					3: '1 Min Volume'}

# Function splits data into ratios as an array, with default 60/20/20
def split(data, ratio = [0.6, 0.2, 0.2]):
	length = len(data)
	train = data[:math.floor(length * ratio[0])]
	valid = data[math.floor(length * ratio[0]):math.floor(length * (ratio[0] + ratio[1]))]
	test = data[math.floor(length * (ratio[0] + ratio[1])):]
	return np.asarray(train), np.asarray(valid), np.asarray(test)


def fit_and_print_acc(model, X, Y, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, score = accuracy_score):

	model.fit(np.concatenate((X_train, X_valid)), np.concatenate((Y_train, Y_valid)))
	acc_train = accuracy_score(Y_train, model.predict(X_train))
	acc_valid = accuracy_score(Y_valid, model.predict(X_valid))
	acc = accuracy_score(Y_test, model.predict(X_test))
	
	print("Train Accuracy: ", acc_train)
	print("Valid Accuracy: ", acc_valid)
	print("Test  Accuracy: ", acc)
	print('Hyperparams:', model.best_params_)

def fit_and_print_reg(model, X, Y, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, method = explained_variance_score, draw = False):

	model.fit(np.concatenate((X_train, X_valid)), np.concatenate((Y_train, Y_valid)))
	metric_train = method(Y_train, model.predict(X_train))
	metric_valid = method(Y_valid, model.predict(X_valid))
	metric = method(Y_test, model.predict(X_test))
	
	metric_string = ''
	if method == explained_variance_score: metric_string = 'Explained Variance' 
	if method == r2_score: metric_string = 'R2'
	print("Train "+metric_string+": ", metric_train)
	print("Valid "+metric_string+": ", metric_valid)
	print("Test  "+metric_string+": ", metric)
	print('Hyperparams:', model.best_params_)

	if draw:
		plt.plot(Y_test, 'g')
		plt.plot(model.predict(X_test), 'r')
		plt.show()


# generates a PredefinedSplit object for GridSearchCV
def initialize_test_fold(X_train, X_valid): 
	test_fold = np.zeros(X_train.shape[0] + X_valid.shape[0])
	for i in range(X_train.shape[0]): test_fold[X_train.shape[0]]     = -1
	for i in range(X_valid.shape[0]): test_fold[X_train.shape[0] + i] = 1
	ps = PredefinedSplit(test_fold)
	return ps

# extracts column and sends values to 1 if > 0, 0 otherwise
def classification_format(Y, col):
	Y_col = [ Y[i][col] for i in range(len(Y))]
	Y_col_arr = np.zeros(len(Y_col))
	for i in range(len(Y_col)):
		if Y_col[i] > 0: Y_col_arr[i] = 1
		else: Y_col_arr[i] = 0
	return Y_col_arr

def multi_classification_format(Y, col):

	# 3 class = 0 BUY, 1 SELL, 2 HOLD
	# fully functional but unused
	# 1 pip = cost of transaction
	# for any meaningful trade you would like at least a double pip jump

	pip = 3.415
	spread = 2.0

	Y_col = [ Y[i][col] for i in range(len(Y))]
	Y_col_arr = np.zeros(len(Y_col))
	for i in range(len(Y_col)):
		if Y_col[i] > spread * pip: Y_col_arr[i] = 0
		elif Y_col[i] < -(spread * pip): Y_col_arr[i] = 1
		else: Y_col_arr[i] = 2
	return Y_col_arr

# extracts raw float columns
def regression_format(Y, col, threshold = 100):
	return np.array([min(threshold, Y[i][col]) for i in range(len(Y))])

# removes first column (text is only in X[i][0])
def remove_text(X, method):
	X_no_text = np.zeros((len(X), len(X[0]) - 1)) 
	for i in range(len(X)):
		for j in range(len(X[0]) - 1):
			if method == 'direct' and j == 0: continue
			X_no_text[i][j] = X[i][j+1]
	return X_no_text 

def vectorize_text(X, method, vocab_length = None, min_df = 1):
	lines = []
	for i in range(len(X)):
		lines.append(X[i][0])

	vectorizer  = CountVectorizer(	analyzer = 'word', 
									binary = True, 
									stop_words = {'english'}, 
									ngram_range = (1,1), 
									min_df = min_df,
									max_features = vocab_length,
									#dtype = np.int32
									)
	vector = vectorizer.fit_transform(lines).toarray()

	X_remaining = np.zeros((len(X), len(X[0]) - 1))
	for i in range(len(X)):
		for j in range(len(X[0]) - 1): 
			if method == 'direct' and j == 0: continue
			X_remaining[i][j] = X[i][j + 1] 

	combined_vector = np.hstack((vector, X_remaining))
	return combined_vector

def eval_classification(X, Y, cols, multi = False):

	X_train, X_valid, X_test = split(X)
	ps = initialize_test_fold(X_train, X_valid)
	score = 'accuracy' # classification
	
	for col in cols:

		Y_col = None
		if multi: Y_col = multi_classification_format(Y, col)
		else: Y_col = classification_format(Y, col)
		Y_train, Y_valid, Y_test = split(Y_col)
		
		# Random Classifier
		print('\nDummy for', col_names[col])
		DUM = DummyClassifier(strategy='most_frequent') 
		DUM.fit(X_train, Y_train) # 
		accuracy = accuracy_score(Y_test, DUM.predict(X_test)) 
		print("Test Accuracy: ", accuracy)
		
		# Linear SVM
		#'''
		print('\nLinear SVC for',col_names[col])
		SVM_parameters = [{ 'tol': [0.0001, 0.001],
							'C': [0.0001, 0.001, 0.01, 0.1] }]
		SVM = GridSearchCV(LinearSVC(), SVM_parameters, cv=ps, scoring=score)
		fit_and_print_acc(SVM, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
		#'''

		# Naive Bayes
		print('\nNaive Bayes for',col_names[col]) 
		
		#NB_parameters = [{'alpha': smoothing}]
		NB_parameters = [{'alpha': [1.0]}]
		NB_model = BernoulliNB()
		if multi: NB_model = GaussianNB()
		NB = GridSearchCV(BernoulliNB(), NB_parameters, cv=ps, scoring=score)
		fit_and_print_acc(NB, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test)

		# DecisionTreeClassifier  
		print('\nDecision Tree Classifier for',col_names[col])
		DT_parameters = None
		#if col == 0: # do thorough test only for 1 min delta
		if 1 == 0: # no thorough test
			DT_parameters = [{  'criterion': ['entropy', 'gini'], #
								'splitter' : ['random', 'best'], #
								#'min_samples_split': [2, 3, 4],
								#'min_samples_leaf': [1, 2, 3],
								'min_weight_fraction_leaf': [0, 0.001, 0.005, 0.007, 0.01, 0.05, 0.1],
								#'min_weight_fraction_leaf': [0.02, 0.2,0.1],
								#'max_features':['auto','sqrt','log2',None],
								#'random_state':[None, 9],
								#'max_leaf_nodes':[3, 4, 5, None],
								'min_impurity_decrease':[0.0, 0.01, 0.1, 0.5],
								#'class_weight':['balanced', None]
								}]
		else:
			DT_parameters = [{ #'splitter' : ['random', 'best'],
								'splitter' : ['random'],
								#'min_weight_fraction_leaf': [0, 0.001, 0.005, 0.007, 0.01, 0.05, 0.1]
								'min_weight_fraction_leaf': [0.1]
								}]

		DT = GridSearchCV(DecisionTreeClassifier(), DT_parameters, cv=ps, scoring=score)
		fit_and_print_acc(DT, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test)

		# LogisticRegression
		#'''
		print('\nLogistic Regression Classifier for', col_names[col])
		# For simplicity sake, use SVM parameters for GridSearch since names are same
		#LR_parameters = [{ 'tol': [0.0001],
		#					'C': [0.0001, 0.001,0.01, 0.1, 0.5, 0.8, 1, 2] }]
		LR = GridSearchCV(LogisticRegression(), SVM_parameters, cv=ps, scoring=score)
		fit_and_print_acc(LR, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
		#'''

def eval_regression(X, Y, cols):
	X_train, X_valid, X_test = split(X)
	ps = initialize_test_fold(X_train, X_valid)

	# alt scoring method
	score = 'explained_variance' 
	method = explained_variance_score

	# regression 2
	score = 'r2' 
	method = r2_score

	for col in cols:

		Y_col = regression_format(Y, col)
		Y_train, Y_valid, Y_test = split(Y_col)

		# Linear SVM
		
		print('\nLinear SVR for', col_names[col])
		SVM_parameters = [{ 'tol': [0.0001],
							'C': [0.0001, 0.001,0.01, 0.1, 0.5, 0.8, 1, 2] }]
		SVM = GridSearchCV(LinearSVR(), SVM_parameters, cv=ps, scoring=score)
		fit_and_print_reg(SVM, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, method = method, draw = False)
		

		# DecisionTreeRegressor
		print('\nDecision Tree Regressor for', col_names[col]) 
		DT_parameters = [{  #'criterion': ['entropy', 'gini'], #
							'splitter' : ['random', 'best'], #
							#'min_samples_split': [2, 3, 4],
							#'min_samples_leaf': [1, 2, 3],
							'min_weight_fraction_leaf': [0, 0.001, 0.005, 0.007, 0.01, 0.05, 0.1],
							#'min_weight_fraction_leaf': [0.02, 0.2,0.1],
							#'max_features':['auto','sqrt','log2',None],
							#'random_state':[None, 9],
							#'max_leaf_nodes':[3,4,5,None],
							'min_impurity_decrease':[0.0, 0.01, 0.1, 0.5],
							#'class_weight':['balanced', None]
							}]

		DT = GridSearchCV(DecisionTreeRegressor(), DT_parameters, cv=ps, scoring=score)
		fit_and_print_reg(DT, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, method = method, draw = False)
		
		'''
		# Ridge 
		print('\nRidge Regression for', col_names[col])
		# For now for simplicity sake, use SVM parameters for GridSearch since names are same
		RIDGE_parameters = [{	'alpha': smoothing, 
								'tol': [0.0001],
								'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] }]
		RIDGE = GridSearchCV(Ridge(), RIDGE_parameters, cv=ps, scoring=score)
		fit_and_print_reg(RIDGE, X, Y_col, X_train, X_valid, X_test, Y_train, Y_valid, Y_test, method = method, draw = False)
		'''

def main():

	method = 'hive' # default
	if len(sys.argv) >= 2:
		method = sys.argv[1]
	
	# load data
	X = pickle.load(open('data/'+method+'_X.pkl',"rb"))
	Y = pickle.load(open('data/'+method+'_Y.pkl',"rb"))

	classification_cols = [0, 1]#, 2] # remove 2? 60 delta is trash
	#regression_cols = [0, 2, 3] # unused but fully functional

	print('"\nCOMPARING PERFORMANCE ON HIVE METHOD"')
	X_no_text = remove_text(X, method) 
	print("**** NO TEXT ****")
	eval_classification(X_no_text, Y, classification_cols)
	X_text = vectorize_text(X,method)
	print("**** TEXT ****")
	eval_classification(X_text, Y, classification_cols)

	method = 'direct'
	print('\n\nCOMPARING PERFORMANCE ON DIRECT METHOD')
	
	X_no_text = pickle.load(open('data/sem/direct_X_no_text.pkl',"rb"))
	X_text = pickle.load(open('data/sem/direct_X_text.pkl',"rb"))
	X_no_text_sem = pickle.load(open('data/sem/direct_X_no_text_sem.pkl',"rb"))
	X_text_sem = pickle.load(open('data/sem/direct_X_text_sem.pkl',"rb"))
	Y = pickle.load(open('data/sem/direct_Y.pkl',"rb"))

	print('\n**** NO TEXT | NO SENTIMENT ****') 
	eval_classification(X_no_text, Y, classification_cols)
	print('\n**** TEXT | NO SENTIMENT ****') 
	eval_classification(X_no_text_sem, Y, classification_cols)
	print('\n**** NO TEXT | SENTIMENT ****')
	eval_classification(X_text, Y, classification_cols)
	print('\n**** TEXT | SENTIMENT ****') 
	eval_classification(X_text_sem, Y, classification_cols)
	

if __name__ == "__main__":
	main()