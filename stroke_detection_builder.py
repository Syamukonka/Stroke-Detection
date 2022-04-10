"""
Authur: Syamukonka Moonga
Title: Stroke detection - using Machine Learning

Approach:
-Use 3 - 4 models and evaluate their individual performance
-Perform feature selection
-Use the Grid Search for hyperparameter tuning
-Use an ensembled model of the models and compare
-Evaluate performance of all approaches and compare

"""

#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve

def plot_overfitting_analysis(  y_test,y_train,y_pred, y_tr_pred, chart_title):
   
    '''Enter the test-set target, training-set target, model prediction on test-set, model prediction on training-set'''
    #COUNT CORRECT AND INCORRECT SAMPLES from test and training set
    correct, incorrect = (y_pred==y_test).sum(), (y_pred!=y_test).sum()
    tr_correct, tr_incorrect = (y_train==y_tr_pred).sum(),(y_train!=y_tr_pred).sum()
    
    #normalise the values between 0 - 1
    correctstd = (correct/(correct + incorrect))*100
    incorrectstd = (incorrect/(correct + incorrect))*100
    
    tr_correctstd = (tr_correct/(tr_correct+tr_incorrect))*100
    tr_incorrectstd = (tr_incorrect/(tr_correct+tr_incorrect))*100

    #SHOW THE PERCENTAGES
    
    
    #MAKE BAR PLOT
    plt.title(chart_title)
    plt.ylabel("Counts")

    plt.bar(["Test set"],[correctstd,incorrectstd], color = ['#8DC6FF','#F9839C'])
    plt.bar(["Train set"],[tr_correctstd,tr_incorrectstd], color = ['#8DC6FF','#F9839C'])
    
    #SHOW MISCLASSIFICATION COUNT
    print("\nMISCLASSIFIED SAMPLES")
    print('Test set: %d --> %.2f%%'%(incorrect, incorrectstd))#compute
    print('Train set: %d --> %.2f%%'%(tr_incorrect, tr_incorrectstd))#compute
    print('\n')
    
    print("CORRECTLY CLASSIFIED SAMPLES")
    print('Test set: %d --> %.2f%%'%(correct, correctstd))#compute
    print('Train set: %d --> %.2f%%'%(tr_correct, tr_correctstd))#compute
    print('\n')    
    
    #SHOW ACCURACY SCORE
    print('Test set Accuracy:%.2f'%accuracy_score(y_test,y_pred))
    print('Train set Accuracy:%.2f'%accuracy_score(y_train,y_tr_pred))
    print('\n')

#retrieve the dataset
raw_dataset = pd.read_csv("C:\\Users\\bossm\\OneDrive\\Semester 6\\INT-247 Machine Learning Foundation\\StrokeDetection\\healthcare-dataset-stroke-data.csv")
#print(raw_dataset.head())


#GET TRAINING AND TARGET DATA
x = raw_dataset.iloc[:,:11] #all rows and all column before col 11"""
y = raw_dataset.iloc[:,11] #all rows of column 11"""

#CHECK UNIQUE CATEGORICAL VALUES 
print("\nUnique categorical values\n\nGender Values\n", x['gender'].unique())
print("ever-married Values\n", x['ever_married'].unique())
print("work-type Values\n", x['work_type'].unique())
print("Residence-type Values\n", x['Residence_type'].unique())
print("smoking-status values\n", x['smoking_status'].unique())
print("stroke Values\n", y.unique())

#CHECK MISSING VALUES
print("\nMissing Values per column:\n",x.isna().sum())

#DATA PROPROCCESSING 1 - IMPUTING MISSING DATA
print("BMI missing values: ",x['bmi'].isna().sum())
avg = x['bmi'].sum()/(x['bmi'].size-201)
x['bmi'] = x['bmi'].fillna(avg)
print("BMI missing value: ",x['bmi'].isna().sum())

#print(x['gender'].value_counts())


#DATA PROPROCCESSING 1 - IMPUTING MISSING DATA
#from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#DATA PREPROCESSING 2 - REMOVE THE OBVIOUSLY UNECCESSARY COLUMNS
del x['id']

#DATA PREPROCESSING 3 - ONE HOT ENCODING 
#Using Pandas get dummies
x = pd.get_dummies(x,drop_first=False)
#Remove column for gender 'Other' as there is only one occurance
x.drop('gender_Other',axis=1,inplace=True)

print("\nNew Dataset shape: ",(x.shape))
print(x)

y = y.ravel()


#TRAINING AND TEST SET INITIALIZATION
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.8, random_state=10, stratify=y)

#SCALING DATA
#from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MMS

std_scaler = MMS()
xtrain_std = std_scaler.fit_transform(xtrain)
xtest_std = std_scaler.transform(xtest)

#FITTING MODELS [SVM | KNN | RANDOM-FOREST | DECISION-TREE]


svm_clf = SVC()
tree_clf = DecisionTreeClassifier(max_depth=9)
knn_clf = KNeighborsClassifier()
ranf_clf = RandomForestClassifier(n_estimators=200,max_depth=9)

#STRATEGY 
# - EVALUATE EACH MODEL
# - COMPARE THERE PERFORMANCE
# - ENSEMBLE




"""   ''' ''' '''  """
""" SVM EVALUATION """

"""   ''' ''' '''  """
svm = SVC(C=1)
#set the tunable hyperparams
svm_params = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
              'random_state':[1,2,3,4]}
svm_gridsearch = GridSearchCV(svm,svm_params,cv=5)
svm_gridsearch.fit(xtrain_std, ytrain)
svm_best = svm_gridsearch.best_estimator_

print("\nSVM PARAMETER TUNING\nBEST PARAMS -",svm_gridsearch.best_params_)

svm_test_pred = svm_best.predict(xtest_std)  #predict on the test set
svm_train_pred = svm_best.predict(xtrain_std) #predict on the training set

plot_overfitting_analysis(ytest,ytrain,svm_test_pred,svm_train_pred,"SVM ANALYSIS")



"""  '' ''' ''' ''' ''' ''' ''' '''  """
''' K-NEAREST-NEIGHBORS EVALUATION '''

"""  '' ''' ''' ''' ''' ''' ''' '''  """

knn = KNeighborsClassifier()
knn_params = {'n_neighbors':range(1,25),'p':[1,2]}
knn_gridsearch = GridSearchCV(knn, knn_params, cv=5)
knn_gridsearch.fit(xtrain_std, ytrain)
knn_best = knn_gridsearch.best_estimator_

print("\nKNN PARAMETER TUNING\nBEST PARAMS -",knn_gridsearch.best_params_)

knn_test_pred = knn_best.predict(xtest_std)
knn_train_pred=knn_best.predict(xtrain_std)

plot_overfitting_analysis(ytest , ytrain , knn_test_pred, knn_train_pred, "KNN AVALYSIS")




"""  '' ''' ''' ''' ''' '''  """
""" RANDOM-FOREST EVALUATION """

"""  '' ''' ''' ''' ''' '''  """

ranf = RandomForestClassifier()
ranf_params = {'n_estimators': [50, 100, 200]}
ranf_gridsearch = GridSearchCV(ranf, ranf_params, cv=5)
ranf_gridsearch.fit(xtrain_std, ytrain)

ranf_best = ranf_gridsearch.best_estimator_
print("\nRANDOM-FOREST PARAMETER TUNING\nBEST PARAMS -",ranf_gridsearch.best_params_)

ranf_test_pred = ranf_best.predict(xtest_std)
ranf_train_pred = ranf_best.predict(xtrain_std)

plot_overfitting_analysis(ytest, ytrain, ranf_test_pred, ranf_train_pred, "RANDOM FOREST AVALYSIS")



"""  '' ''' ''' ''' ''' '''  """
""" DECISION-TREE EVALUATION """

"""  '' ''' ''' ''' ''' '''  """

tree = DecisionTreeClassifier()
tree_params = {'max_depth':range(2,25),'criterion':['gini','entropy'],'splitter':["best", "random"]}
tree_gridsearch = GridSearchCV(tree, tree_params, cv=5)
tree_gridsearch.fit(xtrain_std,ytrain)

tree_best = tree_gridsearch.best_estimator_
print("\nDECISION-TREE PARAMETER TUNING\nBEST PARAMS - ",tree_gridsearch.best_params_)

tree_test_pred = tree_best.predict(xtest_std)
tree_train_pred = tree_best.predict(xtrain_std)

plot_overfitting_analysis(ytest, ytrain, tree_test_pred, tree_train_pred, "DECISION-TREE AVALYSIS")


print('knn: {}'.format(knn_best.score(xtest_std, ytest)))
print('ranf: {}'.format(ranf_best.score(xtest_std, ytest)))
print('tree: {}'.format(tree_best.score(xtest_std, ytest)))
print('svm: {}'.format(svm_best.score(xtest_std, ytest)))


MLA  = [svm_best, knn_best, ranf_best, tree_best]

MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:  
    
    predicted = alg.fit(xtrain_std, ytrain).predict(xtest_std)
    fp, tp, th = roc_curve(ytest, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA used'] = MLA_name
    MLA_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(xtrain_std, ytrain), 4)
    MLA_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(xtest_std, ytest), 4)
    MLA_compare.loc[row_index, 'Precision'] = precision_score(y_true=ytest, y_pred=predicted)
    MLA_compare.loc[row_index, 'Recall'] = recall_score(ytest, predicted)
    MLA_compare.loc[row_index, 'AUC'] = auc(fp, tp)

    row_index+=1
    
MLA_compare.sort_values(by = ['Train Accuracy'], ascending = False, inplace = True)    


"""  '' ''' ''' ''' ''' '''  """
""" ''' VOTING CLASSIFIER '' """

"""  '' ''' ''' ''' ''' '''  """
from sklearn.ensemble import VotingClassifier

#create a dictionary of our models
estimators=[('knn', knn_best), ('ranf', ranf_best), ('tree', tree_best), ('svm', svm_best)]

#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')
ensemble1 = BaggingClassifier(bootstrap=True)
ensemble1.estimators_ = estimators
#fit model to training data
ensemble.fit(xtrain_std, ytrain)
ensemble1.fit(xtrain_std, ytrain)

print("\nENSEMBLE SCORE")
print(ensemble.score(xtest_std, ytest))

print("\nENSEMBLE 2 SCORE")
print(ensemble1.score(xtest_std, ytest))












 

















