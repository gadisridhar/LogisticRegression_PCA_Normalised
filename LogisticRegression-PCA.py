#!/usr/bin/env python
# coding: utf-8

# In[7]:


############################ For regression: f_regression, mutual_info_regression
############################ For classification: chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, f_classif, mutual_info_classif, mutual_info_regression
from time import time
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, auc, classification_report, f1_score, plot_roc_curve, roc_auc_score, roc_curve


# In[8]:


df = pd.read_csv('cancer.csv')
df.head()


# In[9]:


df.describe()


# In[10]:


df.columns


# In[11]:


df['diagnosis'].replace(['M','B'],[1,0], inplace=True)


# In[12]:


df.tail()


# In[13]:


df = df[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','smoothness_se','compactness_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','symmetry_worst','fractal_dimension_worst','diagnosis']]


# In[14]:


df.tail()


# In[15]:


#chekcing class category count
df['diagnosis'].value_counts()


# In[16]:


df['radius_mean'] = np.log(df['radius_mean'])
df['texture_mean'] = np.log(df['texture_mean'])
df['perimeter_mean'] = np.log(df['perimeter_mean'])
df['area_mean'] = np.log(df['area_mean'])
df['smoothness_mean'] = np.log(df['smoothness_mean'])
df['compactness_mean'] = np.log(df['compactness_mean'])
df['symmetry_mean'] = np.log(df['symmetry_mean'])
df['fractal_dimension_mean'] = np.log(df['fractal_dimension_mean'])
df['radius_se'] = np.log(df['radius_se'])
df['texture_se'] = np.log(df['texture_se'])
df['perimeter_se'] = np.log(df['perimeter_se'])
df['smoothness_se'] = np.log(df['smoothness_se'])
df['compactness_se'] = np.log(df['compactness_se'])
df['symmetry_se'] = np.log(df['symmetry_se'])
df['fractal_dimension_se'] = np.log(df['fractal_dimension_se'])
df['radius_worst'] = np.log(df['radius_worst'])
df['texture_worst'] = np.log(df['texture_worst'])
df['perimeter_worst'] = np.log(df['perimeter_worst'])
df['area_worst'] = np.log(df['area_worst'])
df['smoothness_worst'] = np.log(df['smoothness_worst'])
df['compactness_worst'] = np.log(df['compactness_worst'])
df['symmetry_worst'] = np.log(df['symmetry_worst'])
df['fractal_dimension_worst'] = np.log(df['fractal_dimension_worst'])


# In[17]:


df.tail()


# In[18]:


X = df.iloc[:, 0:23].values
X


# In[19]:


type(X)


# In[20]:


Y = df.iloc[:, 23].values
Y


# In[21]:


type(Y)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)


# In[25]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(type(X_train))
print(type(Y_train))


# In[26]:


print(X_train)


# In[27]:


scaler = StandardScaler()


# In[28]:


scaler.fit_transform(X_train)


# In[29]:


print(X_train)


# In[30]:


print(X_test)


# In[31]:


scaler.transform(X_test)


# In[32]:


print(X_test)


# In[33]:


pca = PCA(n_components=15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[34]:


pd.DataFrame(X_train)


# In[35]:


pd.DataFrame(X_test)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# In[38]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[39]:


lr.coef_


# In[40]:


Y_Pred_train = lr.predict(X_train)
Y_Pred_test = lr.predict(X_test)


# In[41]:


print(Y_Pred_train.shape)
print(Y_Pred_test.shape)


# In[43]:


acc_Score_train =  accuracy_score(Y_train,Y_Pred_train)
acc_Score_test = accuracy_score(Y_test,Y_Pred_test)
print("TRAIN SCORE : ", acc_Score_train)
print("TEST SCORE : ", acc_Score_test)


# #Attempt 3
# TRAIN SCORE :  0.9597989949748744
# TEST SCORE :  0.9883040935672515

# #Attempt 2
# TRAIN SCORE :  0.9648351648351648
# TEST SCORE :  0.9824561403508771

# #Attempt 1
# TRAIN SCORE :  0.978021978021978
# TEST SCORE :  0.9736842105263158

# In[44]:


cmTrain  = confusion_matrix(Y_train, Y_Pred_train)
print(cmTrain)


# In[45]:


cmTest =  confusion_matrix(Y_test, Y_Pred_test)
print(cmTest)


# In[46]:


classifyTestReport = classification_report(Y_test, Y_Pred_test)
print(classifyTestReport)


# In[47]:


classifyTrainReport = classification_report(Y_train, Y_Pred_train)
print(classifyTrainReport)


# In[48]:


# TEST DATA GETTING TPR(SIMILARITY) AND FPR(1-SPECIFICITY) WITH ROC_CURVE FUNCTION
Y_PredProba_test = lr.predict_proba(X_test)


# In[49]:


fprTest,tprTest,thresholdTest = roc_curve(Y_test, Y_PredProba_test[:,1], pos_label=1)
print(fprTest)
print(tprTest)
print(thresholdTest)


# In[50]:


# TRAIN DATA GETTING TPR AND FPR WITH ROC_CURVE FUNCTION
Y_PredProba_train = lr.predict_proba(X_train)


# In[51]:


fprTrain,tprTrain,thresholdTrain = roc_curve(Y_train, Y_PredProba_train[:,1], pos_label=1)


# In[52]:


print(fprTrain)
print(tprTrain)
print(thresholdTrain)


# In[53]:


#plot roc curves
plt.plot(fprTest, tprTest, linestyle='--',color='red', label='Logistic Regression')
plt.plot(fprTrain, tprTrain, linestyle='--',color='green', label='Logistic Regression')
#plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('TEST DATA - TRAIN DATA - ROC curve')
# x label
plt.xlabel('False Positive Rate - (1-Specificity)')
# y label
plt.ylabel('True Positive rate - (Sensitivity)')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[ ]:





# In[8]:


# print(df['radius_mean'].isnull().sum())
# print(df['texture_mean'].isnull().sum())
# print(df['perimeter_mean'].isnull().sum())
# print(df['area_mean'].isnull().sum())
# print(df['smoothness_mean'].isnull().sum())
# print(df['compactness_mean'].isnull().sum())
# print(df['concavity_mean'].isnull().sum())
# print(df['concave points_mean'].isnull().sum())
# print(df['symmetry_mean'].isnull().sum())
# print(df['fractal_dimension_mean'].isnull().sum())
# print(df['radius_se'].isnull().sum())
# print(df['texture_se'].isnull().sum())
# print(df['perimeter_se'].isnull().sum())
# print(df['area_se'].isnull().sum())
# print(df['smoothness_se'].isnull().sum())
# print(df['compactness_se'].isnull().sum())
# print(df['concavity_se'].isnull().sum())
# print(df['concave points_se'].isnull().sum())
# print(df['symmetry_se'].isnull().sum())
# print(df['fractal_dimension_se'].isnull().sum())
# print(df['radius_worst'].isnull().sum())
# print(df['texture_worst'].isnull().sum())
# print(df['perimeter_worst'].isnull().sum())
# print(df['area_worst'].isnull().sum())
# print(df['smoothness_worst'].isnull().sum())
# print(df['compactness_worst'].isnull().sum())
# print(df['concavity_worst'].isnull().sum())
# print(df['concave points_worst'].isnull().sum())
# print(df['symmetry_worst'].isnull().sum())
# print(df['fractal_dimension_worst'].isnull().sum())
# print(df['diagnosis'].isnull().sum())


# In[11]:


#before transformation
plt.figure(figsize=(5,5))
plt.hist(df['radius_mean'])


# In[14]:


#after transformation
trm = df['radius_mean']
trm = np.log(trm)
plt.figure(figsize=(5,5))
plt.hist(trm)


# In[68]:


print(trm.mean())
print(trm.std())


# In[12]:


#before transformation
plt.figure(figsize=(5,5))
plt.hist(df['texture_mean'])


# In[15]:


#after transformation
ttm = df['texture_mean']
ttm = np.log(ttm)
plt.figure(figsize=(5,5))
plt.hist(ttm)


# In[16]:


#before transformation
plt.figure(figsize=(5,5))
plt.hist(df['perimeter_mean'])


# In[17]:


#after transformation
tpm = df['perimeter_mean']
tpm = np.log(tpm)
plt.figure(figsize=(5,5))
plt.hist(tpm)


# In[18]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['area_mean'])
tam = df['area_mean']
tam = np.log(tam)
plt.figure(figsize=(5,5))
plt.hist(tam)


# In[19]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['smoothness_mean'])
tsm = df['smoothness_mean']
tsm = np.log(tsm)
plt.figure(figsize=(5,5))
plt.hist(tsm)


# In[21]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['compactness_mean'])
tcm = df['compactness_mean']
tcm = np.log(tcm)
plt.figure(figsize=(5,5))
plt.hist(tcm)


# In[27]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['concavity_mean'])
tconm = df['concavity_mean']
tconm = np.log(tconm)
plt.figure(figsize=(5,5))
plt.hist(tconm)


# In[31]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['concave points_mean'])
# tcpm = df['concave points_mean']
# tcpm = np.log(tcpm)
# plt.figure(figsize=(5,5))
# plt.hist(tcpm)


# In[30]:


tcpm


# In[32]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['symmetry_mean'])
tsm = df['symmetry_mean']
tsm = np.log(tsm)
plt.figure(figsize=(5,5))
plt.hist(tsm)


# In[33]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['fractal_dimension_mean'])
tfdm = df['fractal_dimension_mean']
tfdm = np.log(tfdm)
plt.figure(figsize=(5,5))
plt.hist(tfdm)


# In[34]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['radius_se'])
trse = df['radius_se']
trse = np.log(trse)
plt.figure(figsize=(5,5))
plt.hist(trse)


# In[35]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['texture_se'])
ttse = df['texture_se']
ttse = np.log(ttse)
plt.figure(figsize=(5,5))
plt.hist(ttse)


# In[36]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['perimeter_se'])
tpse = df['perimeter_se']
tpse = np.log(tpse)
plt.figure(figsize=(5,5))
plt.hist(tpse)


# In[37]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['area_se'])
tase = df['area_se']
tase = np.log(tase)
plt.figure(figsize=(5,5))
plt.hist(tase)


# In[38]:


#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['smoothness_se'])
tsse = df['smoothness_se']
tsse = np.log(tsse)
plt.figure(figsize=(5,5))
plt.hist(tsse)


# In[39]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['compactness_se'])
tcose = df['compactness_se']
tcose = np.log(tcose)
plt.figure(figsize=(5,5))
plt.hist(tcose)


# In[40]:




#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['concavity_se'])
tconse = df['concavity_se']
tconse = np.log(tconse)
plt.figure(figsize=(5,5))
plt.hist(tconse)


# In[41]:




#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['concave points_se'])
tconpse = df['concave points_se']
tconpse = np.log(tconpse)
plt.figure(figsize=(5,5))
plt.hist(tconpse)


# In[42]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['symmetry_se'])
tsyse = df['symmetry_se']
tsyse = np.log(tsyse)
plt.figure(figsize=(5,5))
plt.hist(tsyse)


# In[43]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['fractal_dimension_se'])
tfdse = df['fractal_dimension_se']
tfdse = np.log(tfdse)
plt.figure(figsize=(5,5))
plt.hist(tfdse)


# In[44]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['radius_worst'])
trw = df['radius_worst']
trw = np.log(trw)
plt.figure(figsize=(5,5))
plt.hist(trw)


# In[45]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['texture_worst'])
ttw = df['texture_worst']
ttw = np.log(ttw)
plt.figure(figsize=(5,5))
plt.hist(ttw)


# In[46]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['perimeter_worst'])
tpw = df['perimeter_worst']
tpw = np.log(tpw)
plt.figure(figsize=(5,5))
plt.hist(tpw)


# In[47]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['area_worst'])
taw = df['area_worst']
taw = np.log(taw)
plt.figure(figsize=(5,5))
plt.hist(taw)


# In[48]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['smoothness_worst'])
tsmw = df['smoothness_worst']
tsmw = np.log(tsmw)
plt.figure(figsize=(5,5))
plt.hist(tsmw)


# In[49]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['compactness_worst'])
tcmw = df['compactness_worst']
tcmw = np.log(tcmw)
plt.figure(figsize=(5,5))
plt.hist(tcmw)


# In[50]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['concavity_worst'])
tcnw = df['concavity_worst']
tcnw = np.log(tcnw)
plt.figure(figsize=(5,5))
plt.hist(tcnw)


# In[51]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['concave points_worst'])
tcnpw = df['concave points_worst']
tcnpw = np.log(tcnpw)
plt.figure(figsize=(5,5))
plt.hist(tcnpw)


# In[52]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['symmetry_worst'])
tsymw = df['symmetry_worst']
tsymw = np.log(tsymw)
plt.figure(figsize=(5,5))
plt.hist(tsymw)


# In[53]:



#before and after transformation
plt.figure(figsize=(5,5))
plt.hist(df['fractal_dimension_worst'])
tfdiw = df['fractal_dimension_worst']
tfdiw = np.log(tfdiw)
plt.figure(figsize=(5,5))
plt.hist(tfdiw)


# In[ ]:





# In[ ]:


# print(df['radius_mean'].isnull().sum())
# print(df['texture_mean'].isnull().sum())
# print(df['perimeter_mean'].isnull().sum())
# print(df['area_mean'].isnull().sum())
# print(df['smoothness_mean'].isnull().sum())
# print(df['compactness_mean'].isnull().sum())
# print(df['concavity_mean'].isnull().sum())
# print(df['concave points_mean'].isnull().sum())
# print(df['symmetry_mean'].isnull().sum())
# print(df['fractal_dimension_mean'].isnull().sum())
# print(df['radius_se'].isnull().sum())
# print(df['texture_se'].isnull().sum())
# print(df['perimeter_se'].isnull().sum())
# print(df['area_se'].isnull().sum())
# print(df['smoothness_se'].isnull().sum())
# print(df['compactness_se'].isnull().sum())
# print(df['concavity_se'].isnull().sum())
# print(df['concave points_se'].isnull().sum())
# print(df['symmetry_se'].isnull().sum())
# print(df['fractal_dimension_se'].isnull().sum())
# print(df['radius_worst'].isnull().sum())
# print(df['texture_worst'].isnull().sum())
# print(df['perimeter_worst'].isnull().sum())
# print(df['area_worst'].isnull().sum())
# print(df['smoothness_worst'].isnull().sum())
# print(df['compactness_worst'].isnull().sum())
# print(df['concavity_worst'].isnull().sum())
# print(df['concave points_worst'].isnull().sum())
# print(df['symmetry_worst'].isnull().sum())
# print(df['fractal_dimension_worst'].isnull().sum())

