#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[3]:


#import required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("C:/Users/Administrator/Downloads/pronostico_dataset.csv",sep =";")


# In[242]:


# Checking first 5 records to get the idea about the dataset
df.head()


# In[243]:


# Checking last 5 records
df.tail()


# In[244]:


# finding the shape of the dataframe 

df.shape


# ### 6000 Observations with 6 Variables are present in out dataset

# In[245]:


#info of the dataset
df.info()


# ### Missing values Imputation is not required as there are no missing values in the dataset

# In[246]:


#describing the dataset

df.describe()


# In[247]:


#checking the number of null values

df.isnull().sum()


# ### no null values are present in the dataset 

# In[248]:


#checking the datatypes of the variables

df.dtypes


# ### Data Type Conversion is not required as the Data Types of all the columns is in required format except prognosis column

# In[249]:


#checking whether there are any duplicated values

df[df.duplicated()]


# ### No duplicate rows are present in the dataset

# In[250]:


#renaming the columns for easy usage

df1 = df.rename({'systolic_bp': 'sysbp',"diastolic_bp":"diabp","cholesterol":"chol","prognosis":"prog"}, axis=1)


# In[251]:


#updated dataset with new names(short forms)

df1


# In[252]:


#removing ID column as it is not required

df2 = df1.drop(["ID"],axis= 1)


# In[253]:


#updated dataset

df2


# In[254]:


#importing for the label encoder

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[255]:


#label encoding the prognosis column

df2["prog"] = label_encoder.fit_transform(df2["prog"])


# In[256]:


#updated dataset

df2


# ### Correlation Analysis

# In[257]:


df2.corr()


# In[258]:


#info of the updated dataset

df2.info()

#after label encoding now all the variables are numerical type


# #Age and Prognosis(Which is the target variable)are mostly correlated.

# In[259]:


#heatmap between the correlation of the variables
   
sns.heatmap(df2.corr(),annot=True,fmt=".2f")
plt.show()


# In[260]:


#pairplot of the updated dataset

sns.pairplot(df2)


# ### Outlier Analysis

# In[261]:


#boxplot for the updated dataset leaving prognosis column(all in one single image)

df2.boxplot(column=['age',"sysbp","diabp","chol"],figsize=(10,10))


# In[262]:


#histogram of the prognosis column

df2['prog'].value_counts().plot.bar()


# In[263]:


#histogram of the variables
df2.hist(figsize=(10,8))
plt.show()


# In[264]:


df2.skew()


# #### They are moderately skewed

# In[265]:


#pie chart of the column prognosis
df2['prog'].value_counts().plot.pie()


# In[266]:


# Horizontal boxplot of the datset images except prognosis

plt.figure(figsize=(16,14))
sns.set_style(style='ticks')
plt.subplot(3,3,1)
sns.boxplot(x='age',data=df2)
plt.subplot(3,3,2)
sns.boxplot(x='sysbp',data=df2)
plt.subplot(3,3,3)
sns.boxplot(x='diabp',data=df2)
plt.subplot(3,3,4)
sns.boxplot(x='chol',data=df2)
plt.show()


# ### Descriptive Statistics
# 
# The difference between min - 25% data and 75% - max range can give us a brief idea about outliers and the analysis is as follows

# In[267]:


df2.describe()


# In[268]:


#count value of the prognosis column

df2["prog"].value_counts()


# In[269]:


# Let's keep a copy or data before removing outliers
df3 = df2.copy()
df3


# ### Standardisation of the data

# In[270]:


# Importing required libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[271]:


df4 = df3.drop(["prog"],axis= 1)


# In[272]:


scaler=StandardScaler()

data_scaled=scaler.fit_transform(df4)
std=pd.DataFrame(scaler.fit_transform(df4),columns=df4.columns)


# In[273]:


std


# # Auto_EDA

# In[274]:


import pandas_profiling as pp
import sweetviz as sv


# In[214]:


#EDA_report= pp.ProfileReport(df2)
#EDA_report.to_file(output_file='report.html')


# In[215]:


#sweet_report = sv.analyze(df2)
#sweet_report.show_html('prognosis_report.html')


# # MODEL BUILDING

# # LOGISTIC REGRESSION 

# In[275]:


# RENAMING THE ORIGINAL DATASET FOR LOGISTIC REGRESSION MODEL 
dfL = df3
dfL


# In[278]:


#SEPERATING THE TARGET AND PREDICTOR VARIABLES 
# x = PREDICTORS .
# y = TARGET.""
x = dfL.drop('prog',axis=1)
y = dfL['prog']
x


# In[279]:


y


# In[280]:


#TRAIN TEST SPLITTING

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.50,random_state=100)


# In[281]:


from sklearn.linear_model import LogisticRegression
modellogr = LogisticRegression()
modellogr.fit(X_train,y_train)


# In[282]:


y_pred = modellogr.predict(X_test)


# In[283]:


y_pred = modellogr.predict(X_test)
print('Train Score is : ' , modellogr.score(X_train, y_train))
print('Test Score is : ' , modellogr.score(X_test, y_test))


# In[284]:


pd.DataFrame({"Actual":y_test,"Prediction":y_pred})


# In[285]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[286]:


cmlo = confusion_matrix(y_test,y_pred)
cmlo


# In[287]:


accuracy_score(y_test,y_pred) 


# In[288]:


print(classification_report(y_test,y_pred))


# In[289]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[290]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred)

auc = roc_auc_score(y_test,y_pred)


# In[291]:


import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC & AUC curve")
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# # LOGISTIC REGRESSION WITH STANDARDISED VALUES

# In[292]:


C = std
C


# In[293]:


D = y
D


# In[294]:


C_train,C_test,D_train,D_test = train_test_split(C,D,test_size=0.50,random_state=100)


# In[295]:


from sklearn.linear_model import LogisticRegression
modellr = LogisticRegression()
modellr.fit(C_train,D_train)


# In[296]:


D_pred = modellr.predict(C_test)


# In[297]:


print('Train Score is : ' ,modellr.score(C_train, D_train))
print('Test Score is : ' , modellr.score(C_test, D_test))


# In[298]:


pd.DataFrame({"Actual":D_test,"Prediction":D_pred})


# In[299]:


confum = confusion_matrix(D_test,D_pred)
confum


# In[300]:


accuracy_score(D_test,D_pred)


# LOGISTIC REGRESSION MODEL WITH STANDARDISED VALUES = 75.26%

# In[301]:


print(classification_report(D_test,D_pred))


# # USER TESTING FOR LOGISTIC REGRESSION MODEL

# In[302]:


dfL.head()


# In[303]:


def user_testing(data):
    new = pd.DataFrame(data,index=[0])
    result = modellr.predict(new)[0]
    if result==0:
        print("NO")
    else:
        print("YES")


# In[304]:


data = {"age":62,"sysbp":109,"diabp":79,"chol":126}


# In[305]:


user_testing(data)


# # SUPPORT VECTOR MACHINE

# In[306]:


# RENAMING THE COLUMN FOR USING IN SVM MODEL
dfs = df3
dfs


# In[307]:


#SEPERATING THE TARGET AND PREDICTORS
#x1 = PREDICTORS
#y1 = TARGET.
x1 = dfs.drop('prog',axis=1)
y1 = dfs['prog']
x1


# In[308]:


y1


# In[309]:


#standardisation done 
std


# In[310]:


#RENAMING THE DATASET.
svmstd = std
svmstd


# In[311]:


from sklearn.svm import SVC 
svm = SVC(probability=True) # build the model


# In[312]:


Q_train,Q_test,W_train,W_test = train_test_split(x1,y1,test_size = 0.50, random_state = 0)


# In[313]:


svm.fit(Q_train, W_train)


# In[314]:


SVM_pred = svm.predict(Q_test)


# In[315]:


print('Train Score is : ' , svm.score(Q_train, W_train))
print('Test Score is : ' , svm.score(Q_test, W_test)) 


# In[316]:


print('Classification Report is: \n \n' , classification_report(W_test, SVM_pred ))


# In[317]:


print('The accuracy score is:', accuracy_score(W_test, SVM_pred)) # accuracy score
cm = confusion_matrix(W_test,SVM_pred) # Confusion matrix 
print('\n Confusion matrix \n \n', cm)
print(classification_report(W_test,SVM_pred )) #75.5%


# In[318]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[319]:


S_train,S_test,V_train,V_test = train_test_split(x1,y1,test_size = 0.25, random_state = 0)


# In[320]:


model_linear0 = SVC(kernel = "linear")
model_linear0.fit(S_train,V_train)
pred_test_linear0 = model_linear0.predict(S_test)


# In[322]:


np.mean(pred_test_linear0==V_test)*100 


# In[323]:


print('Train Score is : ' ,model_linear0 .score(S_train, V_train))
print('Test Score is : ' , model_linear0 .score(S_test, V_test))


# In[324]:


model_poly0 = SVC(kernel = "poly")
model_poly0.fit(S_train,V_train)
pred_test_poly0 = model_poly0.predict(S_test)


# In[325]:


np.mean(pred_test_poly0==V_test)*100 


# In[326]:


print('Train Score is : ' ,model_poly0 .score(S_train, V_train))
print('Test Score is : ' , model_poly0 .score(S_test, V_test))


# In[327]:


model_rbf0 = SVC(kernel = "rbf")
model_rbf0.fit(S_train,V_train)
pred_test_rbf0 = model_rbf0.predict(S_test)


# In[328]:


#kernel = rbf
np.mean(pred_test_rbf0==V_test)*100 


# In[329]:


print('Train Score is : ' ,model_rbf0.score(S_train, V_train))
print('Test Score is : ' , model_rbf0 .score(S_test, V_test))


# # SVM with standardisation values

# In[330]:


xs1 = svmstd
ys1 = df3['prog']
xs1


# In[331]:


ys1


# In[332]:


M_train,M_test,N_train,N_test = train_test_split(xs1,ys1,test_size = 0.25, random_state = 0)


# In[333]:


model_linear1 = SVC(kernel = "linear")
model_linear1.fit(M_train,N_train)
pred_test_linear1 = model_linear1.predict(M_test)


# In[334]:


np.mean(pred_test_linear1==N_test)*100  


# In[335]:


print('Train Score is : ' ,model_linear1.score(M_train, N_train))
print('Test Score is : ' , model_linear1.score(M_test, N_test))


# In[336]:


model_poly1 = SVC(kernel = "poly")
model_poly1.fit(M_train,N_train)
pred_test_poly1 = model_poly1.predict(M_test)


# In[337]:


np.mean(pred_test_poly1==N_test)*100  #Poly Model accuracy = 74.06%


# In[338]:


print('Train Score is : ' ,model_poly1.score(M_train, N_train))
print('Test Score is : ' ,model_poly1.score(M_test, N_test))


# In[339]:


model_rbf1 = SVC(kernel = "rbf")
model_rbf1.fit(M_train,N_train)
pred_test_rbf1 = model_rbf1.predict(M_test)


# In[340]:


#kernel = rbf
np.mean(pred_test_rbf1==N_test)*100 


# In[341]:


print('Train Score is : ' ,model_rbf1.score(M_train, N_train))
print('Test Score is : ' ,model_rbf1.score(M_test, N_test))


# # SVM WITH NORMALISED VALUES

# In[342]:


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)


# In[343]:


predictors = norm_func(x)
predictors


# In[344]:


target = y
target


# In[345]:


A_train,A_test,B_train,B_test = train_test_split(predictors,target,test_size = 0.25,random_state = 0)


# In[346]:


model_linear2 = SVC(kernel = "linear")
model_linear2.fit(A_train,B_train)
pred_test_linear2 = model_linear2.predict(A_test)


# In[347]:


np.mean(pred_test_linear2==B_test)*100 


# In[348]:


print('Train Score is : ' ,model_linear2 .score(A_train, B_train))
print('Test Score is : ' ,model_linear2 .score(A_test, B_test))


# In[349]:


model_poly2 = SVC(kernel = "poly")
model_poly2.fit(A_train,B_train)
pred_test_poly2 = model_poly2.predict(A_test)


# In[350]:


np.mean(pred_test_poly2==B_test)*100 


# In[351]:


print('Train Score is : ' ,model_poly2.score(A_train, B_train))
print('Test Score is : ' ,model_poly2.score(A_test, B_test))


# In[352]:


# kernel = rbf
model_rbf2 = SVC(kernel = "rbf")
model_rbf2.fit(A_train,B_train)
pred_test_rbf2 = model_rbf2.predict(A_test)


# In[353]:


np.mean(pred_test_rbf2==B_test)*100 


# In[354]:


print('Train Score is : ' ,model_rbf2.score(A_train, B_train))
print('Test Score is : ' ,model_rbf2.score(A_test, B_test))


# # DECISION TREE

# In[355]:


dfD = df3
dfD


# In[356]:


E = dfD.drop('prog',axis=1)
F = dfD['prog']
E


# In[357]:


F


# In[358]:


# Splitting data into train & test
Etrain, Etest, Ftrain, Ftest = train_test_split(E, F, test_size=0.20, random_state=0)


# In[359]:


from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[360]:


model6=DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model6.fit(Etrain,Ftrain)


# In[361]:


DECPRED =model6.predict(Etest)


# In[362]:


print('Train Score is : ' ,model6.score(Etrain, Ftrain))
print('Test Score is : ' ,model6.score(Etest, Ftest))


# In[363]:


pd.Series(DECPRED).value_counts


# In[364]:


np.mean(DECPRED==Ftest)*100 


# In[365]:


print(classification_report(DECPRED,Ftest))


# In[366]:


fn=['age','sysbp','diabp','chol']
cn=['1', '0']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model6,
               feature_names = fn, 
               class_names=cn,
               filled = False);


# # RANDOM FOREST CLASSIFIER

# In[367]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score


# In[376]:


Kfold =    KFold(n_splits=10)
modelrfc =   RandomForestClassifier(n_estimators=50,max_features=3)
resultsrfc=   cross_val_score(modelrfc,E,F,cv=Kfold)
print(resultsrfc.mean())                            


# In[377]:


modelrfc.fit(Etrain, Ftrain)


# In[378]:


rfc_pred = modelrfc.predict(Etest)


# In[379]:


print('Train Score is : ' ,modelrfc.score(Etrain, Ftrain))
print('Test Score is : ' ,modelrfc.score(Etest, Ftest))


# In[380]:


model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[381]:


model_gini.fit(Etrain, Ftrain)


# In[382]:


#Prediction and computing the accuracy
predrfc=model_gini.predict(Etest)
np.mean(DECPRED==Ftest)                   


# # DECISION TREE WITH STANDARDISED VALUES
# 

# In[383]:


std


# In[384]:


decstd = std


# In[385]:


decsE= decstd
decsF= F
decsE


# In[386]:


decsF


# In[387]:


# Splitting data into train & test
decsEtrain, decsEtest, decsFtrain, decsFtest = train_test_split(decsE, decsF, test_size=0.2, random_state=0)


# In[388]:


model67=DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model67.fit(decsEtrain,decsFtrain)


# In[389]:


decs =model67.predict(decsEtest)


# In[390]:


print('Train Score is : ' ,model67.score(decsEtrain, decsFtrain))
print('Test Score is : ' ,model67.score(decsEtest,decsFtest))


# In[391]:


pd.Series(decs).value_counts


# In[392]:


np.mean(decs==decsFtest)*100    


# In[393]:


print(classification_report(decs,decsFtest))


# In[394]:


model_gini2 = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[395]:


model_gini2.fit(decsEtrain, decsFtrain)


# In[396]:


#Prediction and computing the accuracy
predsgini2=model67.predict(decsEtest)
np.mean(predsgini2== decsFtest)             


# # ENSEMBLE TECHNIQUES

# # Bagged Decision Trees for Classification

# In[397]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
filename = "pronostico_dataset seperated.csv"

names = ['age''systolic_bp''diastolic_bp''cholesterol''prognosis']
dataframe = read_csv(filename, names=names)
array = dataframe.values
EN =dfL.drop('prog',axis=1)
TEC = dfL['prog']
seed = 4

kfold = KFold(n_splits=10)
cart = DecisionTreeClassifier()
num_trees = 100
modelENTEC = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)
resultsBDT = cross_val_score(modelENTEC, EN, TEC, cv=kfold)
print(resultsBDT.mean())


# In[398]:


resultsBDT


# # Random Forest Classification

# In[399]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

RA =dfL.drop('prog',axis=1)
FO = dfL['prog']
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=None)
modelRF = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
resultsRF = cross_val_score(modelRF,RA,FO, cv=kfold)
print(resultsRF.mean())


# # AdaBoost Classification

# In[400]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
filename = "pronostico_dataset seperated.csv"

names = ['age''systolic_bp''diastolic_bp''cholesterol''prognosis']
dataframe = read_csv(filename, names=names)
array = dataframe.values

AD=dfL.drop('prog',axis=1)
BO = dfL['prog']

num_trees = 10
seed=7
kfold = KFold(n_splits=10, random_state=None)
modelADBO = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
resultsADBO = cross_val_score(modelADBO,AD,BO, cv=kfold)
print(resultsADBO.mean())


# # Stacking Ensemble for Classification

# In[401]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
filename = "pronostico_dataset seperated.csv"

names = ['age''systolic_bp''diastolic_bp''cholesterol''prognosis']
dataframe = read_csv(filename, names=names)
array = dataframe.values
ST=dfL.drop('prog',axis=1)
AC = dfL['prog']
kfold = KFold(n_splits=10, random_state=None)

# create the sub models
estimators = []
model1 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
resultsSTAC = cross_val_score(ensemble, ST,AC, cv=kfold)
print(resultsSTAC.mean())


# # XGBOOSTING 

# In[402]:


import xgboost
from xgboost import XGBClassifier


# In[403]:


modelXGB = XGBClassifier(n_estimators=10,max_depth=3)
modelXGB.fit(X_train,y_train)


# In[404]:


modelXGB.feature_importances_


# In[405]:


y_predXGB = modelXGB.predict(X_test)


# In[406]:


from sklearn.metrics import confusion_matrix
cmXGB = confusion_matrix(y_test,y_predXGB)
cmXGB


# In[407]:


from sklearn.metrics import classification_report,accuracy_score


# In[408]:


accuracy_score(y_test,y_predXGB)*100  


# In[409]:


print('Train Score is : ' ,modelXGB.score(X_train,y_train))
print('Test Score is : ' ,modelXGB.score(X_test,y_test))


# # NAIVE BAYES

# In[410]:


from sklearn.naive_bayes import GaussianNB


# In[411]:


nb = GaussianNB()
nb.fit(X_train,y_train)
acc = nb.score( X_test,y_test)*100
accuracies = {}
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))  #72.80%


# In[412]:


acc


# In[413]:


print('Train Score is : ' ,nb.score(X_train,y_train))
print('Test Score is : ' ,nb.score(X_test,y_test))


# # NAIVE BAYES WITH STANDARDISED VALUES

# In[414]:


nb1 = GaussianNB()
nb1.fit(C_train,D_train)
acc1= nb1.score(C_test,D_test)*100
accuracies['Naive Bayes'] = acc1
print("Accuracy of Naive Bayes: {:.2f}%".format(acc1)) #73.77%


# In[415]:


print('Train Score is : ' ,nb1.score(C_train,D_train))
print('Test Score is : ' ,nb1.score(C_test,D_test))


# # CROSS VALIDATION SCORE  KNN

# In[416]:


from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
import warnings 
warnings.filterwarnings("ignore")
import os
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# In[417]:


#Get cross validation score of K-Nearest Neighbors


# In[418]:


# plotting the decision boundries for the data 

h = .03  # step size in the mesh
n_neighbors = 3 # No of neighbours
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x, y)


# In[419]:


cv_scores = []


# In[420]:


clf = KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(x, y)


# In[421]:


score_knn=cross_val_score(clf, x,y, cv=10)


# In[422]:


print("K-Nearest Neighbors Accuracy: %0.2f (+/- %0.2f) with k value equals to 3" % (score_knn.mean(), score_knn.std() * 2))


# In[423]:


k_values = np.arange(1,20)
train_accuracy = []
test_accuracy = []


# In[424]:


for i, k in enumerate(k_values):
    # k from 1 to 20(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(X_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(X_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(X_test, y_test))


# In[425]:


print('Train Score is : ' ,knn.score(X_train,y_train))
print('Test Score is : ' ,knn.score(X_test,y_test))


# In[426]:


plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('K-Values VS Accuracy graph representation')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)

plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

cv_scores.append(np.max(test_accuracy))  


# # Tuning of Hyperparameters :- Batch Size and Epochs

# In[427]:


#output 

#Best : 0.7471666693687439, using {'batch_size': 10, 'epochs': 10}

#0.7471666693687439,0.010227947574237293 with: {'batch_size': 10, 'epochs': 10}
#0.7420000076293946,0.009755331114356993 with: {'batch_size': 10, 'epochs': 50}
#0.740666675567627,0.006177911772562141 with: {'batch_size': 10, 'epochs': 100}
#0.7425000071525574,0.01208763138197124 with: {'batch_size': 10, 'epochs': 150}
#0.747000002861023,0.01133577323810853 with: {'batch_size': 20, 'epochs': 10}
#0.7441666722297668,0.008611998908792537 with: {'batch_size': 20, 'epochs': 50}
#0.7430000066757202,0.010389621675347668 with: {'batch_size': 20, 'epochs': 100}
#0.7431666731834412,0.008373229806045154 with: {'batch_size': 20, 'epochs': 150}
#0.747000002861023,0.011286657871752015 with: {'batch_size': 40, 'epochs': 10}
#0.7465000033378602,0.00740494623526058 with: {'batch_size': 40, 'epochs': 50}
#0.7413333415985107,0.009153011416163757 with: {'batch_size': 40, 'epochs': 100}
#0.7431666731834412,0.0019293043105353134 with: {'batch_size': 40, 'epochs': 150}


# # Tuning of Hyperparameters:- Learning rate and Drop out rate

# In[428]:


#OUTPUT

#Best : 0.7493333339691162, using {'dropout_rate': 0.2, 'learning_rate': 0.001}

#0.747000002861023,0.010429648618277176 with: {'dropout_rate': 0.0, 'learning_rate': 0.001}
#0.7443333387374877,0.010292921954991669 with: {'dropout_rate': 0.0, 'learning_rate': 0.01}
#0.7390000104904175,0.01195360265150603 with: {'dropout_rate': 0.0, 'learning_rate': 0.1}
#0.7478333353996277,0.01022794757423729 with: {'dropout_rate': 0.1, 'learning_rate': 0.001}
#0.7455000042915344,0.010705647692123503 with: {'dropout_rate': 0.1, 'learning_rate': 0.01}
#0.737500011920929,0.012030043482324624 with: {'dropout_rate': 0.1, 'learning_rate': 0.1}
#0.7493333339691162,0.013878427278865096 with: {'dropout_rate': 0.2, 'learning_rate': 0.001}
#0.7465000033378602,0.009089056801615689 with: {'dropout_rate': 0.2, 'learning_rate': 0.01}
#0.740500009059906,0.011050379097694402 with: {'dropout_rate': 0.2, 'learning_rate': 0.1}


# # Tuning of Hyperparameters:- Activation Function and Kernel Initializer

# In[429]:


#OUTPUT

#Best : 0.7490000009536744, using {'activation_function': 'relu', 'init': 'uniform'}

#0.7490000009536744,0.009907899849549272 with: {'activation_function': 'relu', 'init': 'uniform'}
#0.7473333358764649,0.011418784532161842 with: {'activation_function': 'relu', 'init': 'normal'}
#0.5144999861717224,0.003188518037474172 with: {'activation_function': 'relu', 'init': 'zero'}
#0.7453333377838135,0.009213507854029319 with: {'activation_function': 'tanh', 'init': 'uniform'}
#0.7438333392143249,0.008621669881978735 with: {'activation_function': 'tanh', 'init': 'normal'}
#0.5144999861717224,0.003188518037474172 with: {'activation_function': 'tanh', 'init': 'zero'}
#0.7443333387374877,0.009389701725942936 with: {'activation_function': 'linear', 'init': 'uniform'}
#0.7453333377838135,0.0090921124604015 with: {'activation_function': 'linear', 'init': 'normal'}
#0.5144999861717224,0.003188518037474172 with: {'activation_function': 'linear', 'init': 'zero'}


# # Tuning of Hyperparameter :-Number of Neurons in activation layer

# In[430]:


#OUTPUT

#Best : 0.7455000042915344, using {'neuron1': 8, 'neuron2': 8}

#0.7446666717529297,0.009228569998879508 with: {'neuron1': 4, 'neuron2': 2}
#0.7446666717529297,0.009436916111508273 with: {'neuron1': 4, 'neuron2': 4}
#0.7450000047683716,0.008547246118675184 with: {'neuron1': 4, 'neuron2': 8}
#0.7440000057220459,0.010033268393715438 with: {'neuron1': 8, 'neuron2': 2}
#0.7448333382606507,0.00985166840879567 with: {'neuron1': 8, 'neuron2': 4}
#0.7455000042915344,0.009015410293092233 with: {'neuron1': 8, 'neuron2': 8}
#0.7451666712760925,0.009315450191074015 with: {'neuron1': 16, 'neuron2': 2}
#0.7451666712760925,0.009315450191074015 with: {'neuron1': 16, 'neuron2': 4}
#0.7446666717529297,0.009582961868610042 with: {'neuron1': 16, 'neuron2': 8}


# # ANN 

# In[431]:


from keras.models import Sequential
from keras.layers import Dense


# In[432]:


ann_x1 = x1
ann_x1


# In[433]:


ann_y1 = y1
ann_y1


# In[434]:


# create model
modelANN = Sequential()
modelANN.add(Dense(12, input_dim=4, kernel_initializer='uniform', activation='relu'))
modelANN.add(Dense(8, bias_initializer='uniform', activation='relu'))
modelANN.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# In[435]:


# Compile model
modelANN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[436]:


# Fit the model
modelANN.fit(ann_x1,ann_y1, validation_split=0.15, epochs=150, batch_size=10)


# In[437]:


# evaluate the model
scoresANN = modelANN.evaluate(ann_x1,ann_y1)
print("%s: %.2f%%" % (modelANN.metrics_names[1], scoresANN[1]*100))


# # MODEL VALIDATION METHODS

# In[438]:


from sklearn.linear_model import LogisticRegression
modelval = LogisticRegression()
modelval.fit(X_train, y_train)
resultval = modelval.score(X_test, y_test)


# In[439]:


resultval


# In[440]:


resultval*100

