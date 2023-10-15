#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[15]:


# file paths for the datasets
train_path = 'D:/UOM/Semester_7/3 - CS4622 - Machine Learning/Kaggle/layer 9/train.csv'
valid_path = 'D:/UOM/Semester_7/3 - CS4622 - Machine Learning/Kaggle/layer 9/valid.csv'
test_path = 'D:/UOM/Semester_7/3 - CS4622 - Machine Learning/Kaggle/layer 9/test.csv'

# Load the train dataset
train_data = pd.read_csv(train_path)

# Load the valid dataset
valid_data = pd.read_csv(valid_path)

# Load the test dataset
test_data = pd.read_csv(test_path)


# In[16]:


# get list of IDs in test dataset
ID = test_data['ID'].to_numpy()

# drop the ID column
test_data = test_data.drop(columns=['ID'])


# In[17]:


train_data.head()


# In[18]:


# datasets for each label in train dataset
train_data_label1 = train_data.drop(columns=['label_2', 'label_3', 'label_4'])
train_data_label2 = train_data.drop(columns=['label_1', 'label_3', 'label_4'])
train_data_label3 = train_data.drop(columns=['label_1', 'label_2', 'label_4'])
train_data_label4 = train_data.drop(columns=['label_1', 'label_2', 'label_3'])

# datasets for each label in train dataset
valid_data_label1 = valid_data.drop(columns=['label_2', 'label_3', 'label_4'])
valid_data_label2 = valid_data.drop(columns=['label_1', 'label_3', 'label_4'])
valid_data_label3 = valid_data.drop(columns=['label_1', 'label_2', 'label_4'])
valid_data_label4 = valid_data.drop(columns=['label_1', 'label_2', 'label_3'])


# # Label 1

# In[19]:


# check for null values in train dataset
print(f"train_data_label1 shape before : {train_data_label1.shape}")
train_null_counts = train_data_label1.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# drop rows with null values in the target labels for train dataset
train_data_label1 = train_data_label1.dropna(subset=train_data_label1.columns[-1:], how='any')
print(f"train_data_label1 shape after : {train_data_label1.shape}")


# In[20]:


# fill null values with mean in train dataset
train_data_label1 = train_data_label1.fillna(train_data_label1.mean())

# fill null values with mean in valid dataset
valid_data_label1 = valid_data_label1.fillna(valid_data_label1.mean())

# fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())


# In[21]:


#seperate features and target labels

train_features_label1 = train_data_label1.iloc[:, :-1]
train_label1 = train_data_label1.iloc[:, -1]

valid_features_label1 = valid_data_label1.iloc[:, :-1]
valid_label1 = valid_data_label1.iloc[:, -1]

test_features_label1 = test_data


# In[22]:


# plot the distribution of train_label1
labels, counts = np.unique(train_label1, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 1')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 1')
plt.show()


# In[23]:


# standardize the features
scaler = RobustScaler()
standardized_train_features_label1 = scaler.fit_transform(train_features_label1)
standardized_valid_features_label1 = scaler.transform(valid_features_label1)
standardized_test_features_label1 = scaler.transform(test_features_label1)


# In[24]:


# threshold for variance
variance_threshold = 0.97

# apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label1 = pca.fit_transform(standardized_train_features_label1)
pca_valid_features_label1 = pca.transform(standardized_valid_features_label1)
pca_test_features_label1 = pca.transform(standardized_test_features_label1)

# explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label1.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# get the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label1.shape))
# get the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label1.shape))
# get the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label1.shape))


# In[37]:


#calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label1).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()


# In[23]:


# threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# get highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)


# In[26]:


# remove highly correlated features
pca_train_features_label1 = pd.DataFrame(pca_train_features_label1).drop(columns=highly_correlated)
pca_valid_features_label1 = pd.DataFrame(pca_valid_features_label1).drop(columns=highly_correlated)
pca_test_features_label1 = pd.DataFrame(pca_test_features_label1).drop(columns=highly_correlated)

# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label1.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label1.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label1.shape))


# In[27]:


# calculate the correlation matrix between features and train_label1
correlation_with_target = pca_train_features_label1.corrwith(train_label1)

# correlation threshold
correlation_threshold = 0.005

# select features that meet the correlation threshold
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)


# In[28]:


# drop the features with low correlated in train data
pca_train_features_label1 = pca_train_features_label1[highly_correlated_features.index]

# drop the features with low correlated in valid data
pca_valid_features_label1 = pca_valid_features_label1[highly_correlated_features.index]

# drop the features with low correlated in test data
pca_test_features_label1 = pca_test_features_label1[highly_correlated_features.index]


# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label1.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label1.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label1.shape))


# In[70]:


# parameters for hyperparameter tuning of label 1 with SVM

svm_grid_params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf','linear']
}


# search across all different combinations, and use all available cores
rs = HalvingGridSearchCV(
    estimator = SVC(),
    param_grid = svm_grid_params,
    factor = 2,
    cv = 3, verbose=2, n_jobs = -1
)
rs_result = rs.fit(pca_train_features_label1, train_label1)

print(f"best score for SVM : {rs_result.best_score_}")
print(f"best hyper parameters for SVM : {rs_result.best_params_}")


# In[25]:


# define the classification model
model = SVC(C=100, gamma=0.001, kernel='rbf')

# get number of features used in PCA
num_features = pca_train_features_label1.shape[1]
print(f"Number of features: {num_features}\n")


# train the model on the training data
model.fit(pca_train_features_label1, train_label1)

# predict on the train data
y_pred_train_label1 = model.predict(pca_train_features_label1)

# calculate metrics for classification evaluation
accuracy = accuracy_score(train_label1, y_pred_train_label1)
precision = precision_score(train_label1, y_pred_train_label1, average='macro', zero_division=1)
recall = recall_score(train_label1, y_pred_train_label1, average='macro')

print(f"Metrics for SVM on train data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the validation data
y_pred_valid_label1 = model.predict(pca_valid_features_label1)

# calculate metrics for classification evaluation on validation data
accuracy = accuracy_score(valid_label1, y_pred_valid_label1)
precision = precision_score(valid_label1, y_pred_valid_label1, average='macro', zero_division=1)
recall = recall_score(valid_label1, y_pred_valid_label1, average='macro')

print(f"Metrics for SVM on validation data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the test data
y_pred_test_label1 = model.predict(pca_test_features_label1)


# # Label 2

# In[26]:


# check for null values in train dataset
print(f"train_data_label2 shape before : {train_data_label2.shape}")
train_null_counts = train_data_label2.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# drop rows with null values in the target labels for train dataset
train_data_label2 = train_data_label2.dropna(subset=train_data_label2.columns[-1:], how='any')
print(f"train_data_label2 shape after : {train_data_label2.shape}")


# In[27]:


# fill null values with mean in train dataset
train_data_label2 = train_data_label2.fillna(train_data_label2.mean())

# fill null values with mean in valid dataset
valid_data_label2 = valid_data_label2.fillna(valid_data_label2.mean())

# fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())


# In[28]:


#seperate features and target labels
train_features_label2 = train_data_label2.iloc[:, :-1]
train_label2 = train_data_label2.iloc[:, -1].astype('int64')

valid_features_label2 = valid_data_label2.iloc[:, :-1]
valid_label2 = valid_data_label2.iloc[:, -1].astype('int64')

test_features_label2 = test_data


# In[29]:


# plot the distribution of train_label2
labels, counts = np.unique(train_label2, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 2')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 2')
plt.show()


# In[30]:


# create an instance of the RandomOverSampler
over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

# create an instance of the RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# create a pipeline that first applies over-sampling and then under-sampling
sampling_pipeline = Pipeline([
    ('over_sampling', over_sampler),
    ('under_sampling', under_sampler)
])

train_features_label2, train_label2 = sampling_pipeline.fit_resample(train_features_label2, train_label2)


# In[31]:


# standardize the features
scaler = RobustScaler()
standardized_train_features_label2 = scaler.fit_transform(train_features_label2)
standardized_valid_features_label2 = scaler.transform(valid_features_label2)
standardized_test_features_label2 = scaler.transform(test_features_label2)


# In[32]:


# threshold for variance
variance_threshold = 0.97

# apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label2 = pca.fit_transform(standardized_train_features_label2)
pca_valid_features_label2 = pca.transform(standardized_valid_features_label2)
pca_test_features_label2 = pca.transform(standardized_test_features_label2)

# explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label2.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# get the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label2.shape))
# get the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label2.shape))
# get the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label2.shape))


# In[79]:


#calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label2).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()


# In[80]:


# threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# get highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)


# In[81]:


# remove highly correlated features
pca_train_features_label2 = pd.DataFrame(pca_train_features_label2).drop(columns=highly_correlated)
pca_valid_features_label2 = pd.DataFrame(pca_valid_features_label2).drop(columns=highly_correlated)
pca_test_features_label2 = pd.DataFrame(pca_test_features_label2).drop(columns=highly_correlated)

# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label2.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label2.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label2.shape))


# In[82]:


# calculate the correlation matrix between features and train_label1
correlation_with_target = pca_train_features_label2.corrwith(train_label2)

# set the correlation threshold
correlation_threshold = 0.005

# select features that meet the correlation threshold
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)


# In[83]:


# drop the features with low correlated in train data
pca_train_features_label2 = pca_train_features_label2[highly_correlated_features.index]

# drop the features with low correlated in valid data
pca_valid_features_label2 = pca_valid_features_label2[highly_correlated_features.index]

# drop the features with low correlated in test data
pca_test_features_label2 = pca_test_features_label2[highly_correlated_features.index]


# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label2.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label2.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label2.shape))


# In[96]:


# parameters for hyperparameter tuning of label 2 with SVM

svm_grid_params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf','linear']
}


# search across all different combinations, and use all available cores
rs = HalvingGridSearchCV(
    estimator = SVC(),
    param_grid = svm_grid_params,
    factor = 2,
    cv = 3, verbose=2, n_jobs = -1
)
rs_result = rs.fit(pca_train_features_label2, train_label2)

print(f"best score for SVM : {rs_result.best_score_}")
print(f"best hyper parameters for SVM : {rs_result.best_params_}")


# In[33]:


# define the classification model
model = SVC(C=100, gamma=0.001, kernel='rbf')

# get number of features used in PCA
num_features = pca_train_features_label2.shape[1]
print(f"Number of features: {num_features}\n")


# train the model on the training data
model.fit(pca_train_features_label2, train_label2)

# predict on the train data
y_pred_train_label2 = model.predict(pca_train_features_label2)

# calculate metrics for classification evaluation
accuracy = accuracy_score(train_label2, y_pred_train_label2)
precision = precision_score(train_label2, y_pred_train_label2, average='macro', zero_division=1)
recall = recall_score(train_label2, y_pred_train_label2, average='macro')

print(f"Metrics for SVM on train data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the validation data
y_pred_valid_label2 = model.predict(pca_valid_features_label2)

# calculate metrics for classification evaluation on validation data
accuracy = accuracy_score(valid_label2, y_pred_valid_label2)
precision = precision_score(valid_label2, y_pred_valid_label2, average='macro', zero_division=1)
recall = recall_score(valid_label2, y_pred_valid_label2, average='macro')

print(f"Metrics for SVM on validation data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the test data
y_pred_test_label2 = model.predict(pca_test_features_label2)


# # Label 3

# In[34]:


# check for null values in train dataset
print(f"train_data_label3 shape before : {train_data_label3.shape}")
train_null_counts = train_data_label3.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# drop rows with null values in the target labels for train dataset
train_data_label3 = train_data_label3.dropna(subset=train_data_label3.columns[-1:], how='any')
print(f"train_data_label3 shape after : {train_data_label3.shape}")


# In[35]:


# fill null values with mean in train dataset
train_data_label3 = train_data_label3.fillna(train_data_label3.mean())

# fill null values with mean in valid dataset
valid_data_label3 = valid_data_label3.fillna(valid_data_label3.mean())

# fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())


# In[36]:


train_features_label3 = train_data_label3.iloc[:, :-1]
train_label3 = train_data_label3.iloc[:, -1]

valid_features_label3 = valid_data_label3.iloc[:, :-1]
valid_label3 = valid_data_label3.iloc[:, -1]

test_features_label3 = test_data


# In[37]:


# plot the distribution of train_label3
labels, counts = np.unique(train_label3, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 3')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 3')
plt.show()


# In[38]:


# create an instance of the RandomOverSampler
over_sampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)

# create an instance of the RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

# create a pipeline that first applies over-sampling and then under-sampling
sampling_pipeline = Pipeline([
    # ('over_sampling', over_sampler),
    ('under_sampling', under_sampler)
])

train_features_label3, train_label3 = sampling_pipeline.fit_resample(train_features_label3, train_label3)


# In[39]:


# standardize the features
scaler = RobustScaler()
standardized_train_features_label3 = scaler.fit_transform(train_features_label3)
standardized_valid_features_label3 = scaler.transform(valid_features_label3)
standardized_test_features_label3 = scaler.transform(test_features_label3)


# In[40]:


# threshold for variance
variance_threshold = 0.95

# apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label3 = pca.fit_transform(standardized_train_features_label3)
pca_valid_features_label3 = pca.transform(standardized_valid_features_label3)
pca_test_features_label3 = pca.transform(standardized_test_features_label3)

# explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label3.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# get the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label3.shape))
# get the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label3.shape))
# get the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label3.shape))


# In[41]:


#calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label3).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()


# In[42]:


# threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# get highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)


# In[43]:


# remove highly correlated features
pca_train_features_label3 = pd.DataFrame(pca_train_features_label3).drop(columns=highly_correlated)
pca_valid_features_label3 = pd.DataFrame(pca_valid_features_label3).drop(columns=highly_correlated)
pca_test_features_label3 = pd.DataFrame(pca_test_features_label3).drop(columns=highly_correlated)

# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label3.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label3.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label3.shape))


# In[44]:


# calculate the correlation matrix between features and train_label3
correlation_with_target = pca_train_features_label3.corrwith(train_label3)

# set the correlation threshold
correlation_threshold = 0.005

# select features that meet the correlation threshold
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)


# In[45]:


# drop the features with low correlated in train data
pca_train_features_label3 = pca_train_features_label3[highly_correlated_features.index]

# drop the features with low correlated in valid data
pca_valid_features_label3 = pca_valid_features_label3[highly_correlated_features.index]

# drop the features with low correlated in test data
pca_test_features_label3 = pca_test_features_label3[highly_correlated_features.index]


# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label3.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label3.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label3.shape))


# In[121]:


# parameters for hyperparameter tuning of label 3 with SVM
svm_grid_params = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf','linear']
}

# search across all different combinations, and use all available cores
rs = HalvingGridSearchCV(
    estimator = SVC(),
    param_grid = svm_grid_params,
    factor = 2,
    cv = 3, verbose=2, n_jobs = -1
)
rs_result = rs.fit(pca_train_features_label3, train_label3)

print(f"best score for SVM : {rs_result.best_score_}")
print(f"best hyper parameters for SVM : {rs_result.best_params_}")


# In[46]:


# define the classification model
model = SVC(C=10, gamma=0.001, kernel='rbf')

# get number of features used in PCA
num_features = pca_train_features_label3.shape[1]
print(f"Number of features: {num_features}\n")

# train the model on the training data
model.fit(pca_train_features_label3, train_label3)

# predict on the train data
y_pred_train_label3 = model.predict(pca_train_features_label3)

# calculate metrics for classification evaluation
accuracy = accuracy_score(train_label3, y_pred_train_label3)
precision = precision_score(train_label3, y_pred_train_label3, average='macro', zero_division=1)
recall = recall_score(train_label3, y_pred_train_label3, average='macro')

print(f"Metrics for SVM on train data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the validation data
y_pred_valid_label3 = model.predict(pca_valid_features_label3)

# calculate metrics for classification evaluation on validation data
accuracy = accuracy_score(valid_label3, y_pred_valid_label3)
precision = precision_score(valid_label3, y_pred_valid_label3, average='macro', zero_division=1)
recall = recall_score(valid_label3, y_pred_valid_label3, average='macro')

print(f"Metrics for SVM on validation data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the test data
y_pred_test_label3 = model.predict(pca_test_features_label3)


# # Label 4

# In[47]:


# check for null values in train dataset
print(f"train_data_label4 shape before : {train_data_label4.shape}")
train_null_counts = train_data_label4.isnull().sum()
print(f"train null counts before : \n{train_null_counts}")

# drop rows with null values in the target labels for train dataset
train_data_label4 = train_data_label4.dropna(subset=train_data_label4.columns[-1:], how='any')
print(f"train_data_label4 shape after : {train_data_label4.shape}")


# In[48]:


# fill null values with mean in train dataset
train_data_label4 = train_data_label4.fillna(train_data_label4.mean())

# fill null values with mean in valid dataset
valid_data_label4 = valid_data_label4.fillna(valid_data_label4.mean())

# fill null values with mean in test dataset
test_data = test_data.fillna(test_data.mean())


# In[49]:


train_features_label4 = train_data_label4.iloc[:, :-1]
train_label4 = train_data_label4.iloc[:, -1]

valid_features_label4 = valid_data_label4.iloc[:, :-1]
valid_label4 = valid_data_label4.iloc[:, -1]

test_features_label4 = test_data


# In[50]:


# plot the distribution of train_label4
labels, counts = np.unique(train_label4, return_counts=True)

plt.figure(figsize=(22, 6))
plt.xticks(labels)
plt.bar(labels, counts)
plt.xlabel('Target Label 4')
plt.ylabel('Frequency')
plt.title('Distribution of Target Label 4')
plt.show()


# In[51]:


# create an instance of the RandomOverSampler
over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

# create an instance of the RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# create a pipeline that first applies over-sampling and then under-sampling
sampling_pipeline = Pipeline([
    ('over_sampling', over_sampler),
    ('under_sampling', under_sampler)
])

train_features_label4, train_label4 = sampling_pipeline.fit_resample(train_features_label4, train_label4)


# In[52]:


# standardize the features
scaler = RobustScaler()
standardized_train_features_label4 = scaler.fit_transform(train_features_label4)
standardized_valid_features_label4 = scaler.transform(valid_features_label4)
standardized_test_features_label4 = scaler.transform(test_features_label4)


# In[53]:


# threshold for variance
variance_threshold = 0.95

# apply PCA with the determined number of components
pca = PCA(n_components=variance_threshold, svd_solver='full')

pca_train_features_label4 = pca.fit_transform(standardized_train_features_label4)
pca_valid_features_label4 = pca.transform(standardized_valid_features_label4)
pca_test_features_label4 = pca.transform(standardized_test_features_label4)

# explained variance ratio after dimensionality reduction
explained_variance_ratio_reduced = pca.explained_variance_ratio_

# plot explained variance ratio
plt.figure(figsize=(18, 10))
plt.bar(range(1, pca_train_features_label4.shape[1] + 1), explained_variance_ratio_reduced)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio per Principal Component (Reduced)')
plt.show()

# get the reduced train feature matrix
print("Reduced Train feature matrix shape: {}".format(pca_train_features_label4.shape))
# get the reduced valid feature matrix
print("Reduced valid feature matrix shape: {}".format(pca_valid_features_label4.shape))
# get the reduced test feature matrix
print("Reduced test feature matrix shape: {}".format(pca_test_features_label4.shape))


# In[21]:


# calculate the correlation matrix
correlation_matrix = pd.DataFrame(pca_train_features_label4).corr()

mask = np.triu(np.ones_like(correlation_matrix))

# create a heatmap of the correlation matrix using seaborn
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask)
plt.title("Correlation Matrix")
plt.show()


# In[22]:


# set the threshold for correlation
correlation_threshold = 0.9

highly_correlated = set()

# get highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated.add(colname)

print(highly_correlated)


# In[23]:


# remove highly correlated features
pca_train_features_label4 = pd.DataFrame(pca_train_features_label4).drop(columns=highly_correlated)
pca_valid_features_label4 = pd.DataFrame(pca_valid_features_label4).drop(columns=highly_correlated)
pca_test_features_label4 = pd.DataFrame(pca_test_features_label4).drop(columns=highly_correlated)

# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label4.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label4.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label4.shape))


# In[24]:


# calculate the correlation matrix between features and train_label4
correlation_with_target = pca_train_features_label4.corrwith(train_label4)

# set the correlation threshold
correlation_threshold = 0.005

# select features that meet the correlation threshold
highly_correlated_features = correlation_with_target[correlation_with_target.abs() > correlation_threshold]

print(highly_correlated_features)


# In[25]:


# drop the features with low correlated in train data
pca_train_features_label4 = pca_train_features_label4[highly_correlated_features.index]

# drop the features with low correlated in valid data
pca_valid_features_label4 = pca_valid_features_label4[highly_correlated_features.index]

# drop the features with low correlated in test data
pca_test_features_label4 = pca_test_features_label4[highly_correlated_features.index]


# get the filtered train feature count
print("Filtered train features: {}".format(pca_train_features_label4.shape))

# get the filtered valid feature count
print("Filtered valid features: {}".format(pca_valid_features_label4.shape))

# get the filtered test feature count
print("Filtered test features: {}".format(pca_test_features_label4.shape))


# In[12]:


# parameters for hyperparameter tuning of label 4 with SVM
svm_grid_params = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf','linear']
}


# search across all different combinations, and use all available cores
rs = HalvingGridSearchCV(
    estimator = SVC(),
    param_grid = svm_grid_params,
    factor = 2,
    cv = 3, verbose=2, n_jobs = -1
)
rs_result = rs.fit(pca_train_features_label4, train_label4)

print(f"best score for SVM : {rs_result.best_score_}")
print(f"best hyper parameters for SVM : {rs_result.best_params_}")


# In[54]:


# define the classification model
model = SVC(C=100, gamma=0.001, kernel='rbf')

# get number of features used in PCA
num_features = pca_train_features_label4.shape[1]
print(f"Number of features: {num_features}\n")

# train the model on the training data
model.fit(pca_train_features_label4, train_label4)

# predict on the train data
y_pred_train_label4 = model.predict(pca_train_features_label4)

# calculate metrics for classification evaluation
accuracy = accuracy_score(train_label4, y_pred_train_label4)
precision = precision_score(train_label4, y_pred_train_label4, average='macro', zero_division=1)
recall = recall_score(train_label4, y_pred_train_label4, average='macro')

print(f"Metrics for SVM on train data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the validation data
y_pred_valid_label4 = model.predict(pca_valid_features_label4)

# calculate metrics for classification evaluation on validation data
accuracy = accuracy_score(valid_label4, y_pred_valid_label4)
precision = precision_score(valid_label4, y_pred_valid_label4, average='macro', zero_division=1)
recall = recall_score(valid_label4, y_pred_valid_label4, average='macro')

print(f"Metrics for SVM on validation data:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\n")

# predict on the test data
y_pred_test_label4 = model.predict(pca_test_features_label4)


# # Generate Output CSV

# Define method to create the csv file

# In[55]:


# define method to create the dataframe and save it as a csv file
def create_csv(ID, pred_label1, pred_label2, pred_label3, pred_label4, destination):
  df = pd.DataFrame()
  
  df.insert(loc=0, column='ID', value=ID)
  df.insert(loc=1, column='label_1', value=pred_label1)
  df.insert(loc=2, column='label_2', value=pred_label2)
  df.insert(loc=3, column='label_3', value=pred_label3)
  df.insert(loc=4, column='label_4', value=pred_label4)

  df.to_csv(destination, index=False)


# Create CSV file

# In[56]:


destination = 'D:/UOM/Semester_7/3 - CS4622 - Machine Learning/Kaggle/layer 9/results/190495.csv'

# create the csv output file
create_csv(ID, y_pred_test_label1, y_pred_test_label2, y_pred_test_label3, y_pred_test_label4, destination)


# In[ ]:




