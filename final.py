import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt, gc, os
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import lightgbm
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay, accuracy_score
import shap
import itertools

#Read Data
train = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/train.parquet").groupby('customer_ID').tail(4)
test = pd.read_parquet("/kaggle/input/amex-data-integer-dtypes-parquet-format/test.parquet").groupby('customer_ID').tail(4)
train_labels = pd.read_csv("../input/amex-default-prediction/train_labels.csv")
train.head()

#Checking shapes of the data
train.shape, test.shape, train_labels.shape

#Data Preprocessing
#Checking Missing values having more than 40% and drop
columns = train.columns[(train.isna().sum()/len(train))*100>40]
train = train.drop(columns, axis=1)
test = test.drop(columns, axis=1)
train = train.bfill(axis='rows').ffill(axis='rows')
test = test.bfill(axis='rows').ffill(axis='rows')
train.reset_index(inplace=True)
test.reset_index(inplace=True)
train =train.groupby('customer_ID').tail(1)
test = test.groupby('customer_ID').tail(1)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

#Check shape
train.shape, train_labels.shape, test.shape

#Change categorical_column type to string
cat_col = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
train[cat_col] = train[obj_col].apply(lambda x: x.astype(str))
test[cat_col] = test[obj_col].apply(lambda x: x.astype(str))

#prepare final data to process further
train = train.merge(train_labels, how='inner', on="customer_ID")
train.head()
test_data = test.copy()
train = train.drop(['index','customer_ID', 'S_2'], axis=1)
test = test.drop(['index','customer_ID', 'S_2'], axis=1)

#feature enginerring
#One-hot encoding
train = pd.get_dummies(train, columns=cat_col, drop_first=True)
test = pd.get_dummies(test, columns=cat_col, drop_first=True)
features=train.loc[:, test.columns]
target = train['target']

#Scaling 
from sklearn.preprocessing import MinMaxScaler
features_columns
minMaxscaler = MinMaxScaler(feature_range=(0, 1))
minMaxscaler.fit(features)
features = scaler.transform(features)
features=pd.DataFrame(features, columns=features_columns)

#outlier handling
import pandas as pd
import numpy as np
outliers_removed_features = pd.DataFrame()
for column in features.columns:
    # calculate the quartiles
    q1, q3 = np.percentile(features[column], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    column_without_outliers = features[column][(features[column] > lower_bound) & (features[column] < upper_bound)]
    outliers_removed_features[column] = column_without_outliers

features=outlier_removed_features

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, target, test_size=0.3, random_state=0)

#XGB Classifier

XGB = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1).fit(Features, target)
np.mean(cross_val_score(XGB, Features, target, scoring='accuracy', cv=5))


y_pred.head()
y_pred = XGB.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(13,5))
plt.title("Confusion Matrix")
plt.imshow(cm, alpha=0.5, cmap='PuBu')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center")
plt.show()


#AUC (ROC curve)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=XGB)
display.plot()
plt.show()





#SVM Clasifier
# Create the SVM classifier
svm_clf = svm.SVC()

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Create a new SVM classifier with the best hyperparameters
best_svm = svm.SVC(**best_params)

# Fit the best model to the data
best_svm.fit(X, y)

y_pred=best_svm.pred(X_test)

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(13,5))
plt.title("Confusion Matrix")
plt.imshow(cm, alpha=0.5, cmap='PuBu')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center")
plt.show()

#AUC (ROC curve)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=XGB)
display.plot()
plt.show()

#Cross validation by Kfold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = XGB, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svm_clf, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()





##Submition data
test_data['prediction']=XGB.predict_proba(test)[:,1]
test_data[['customer_ID','prediction']].to_csv("submission.csv", index=False)



