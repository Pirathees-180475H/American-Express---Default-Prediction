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





