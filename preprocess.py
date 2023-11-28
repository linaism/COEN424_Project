import os
import sklearn
import sklearn.datasets
import sklearn.ensemble
import xgboost
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer, GPTExplainer


credit_record = pd.read_csv('data/credit_record.csv')
application_record = pd.read_csv('data/application_record.csv')

columns = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
for objColumn in columns:
    print(application_record[objColumn])
    label = LabelEncoder()
    application_record[objColumn] = label.fit_transform(application_record[objColumn].values)
    print(application_record[objColumn])

# Drop unnessessary columns
application_record.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True, axis=1)

# Convert categorical to numerical variables and map to either 0 or 1 because it is a binary classification task
# 1 includes users who took no loans that month paid within the month or 30 days past the due date while
# 0 includes users who pay within 30 to 149 days past the due date or have overdue debts for more than 150 days
map_status = {'C' : 1,
              'X' : 1,
              '0' : 1,
              '1' : 0,
              '2' : 0,
              '3' : 0,
              '4' : 0,
              '5' : 0}
credit_record["STATUS"] = credit_record['STATUS'].map(map_status)

# Merge both credit and applications records to create a comprehensive dataset
df_credit = application_record.merge(credit_record, how='inner', on=['ID'])
df_credit.drop(['ID'], inplace=True, axis=1)

CATEGORICAL_FEATURES = [
    "NAME_INCOME_TYPE", 
    "NAME_EDUCATION_TYPE", 
    "NAME_FAMILY_STATUS", 
    "NAME_HOUSING_TYPE", 
    "OCCUPATION_TYPE"
    ]

tabular_data = Tabular(
    data=df_credit,
    categorical_columns=CATEGORICAL_FEATURES,
    target_column='STATUS'
)

# Train an XGBoost model
np.random.seed(1)
transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
x = transformer.transform(tabular_data)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x[:, :-1], x[:, -1], train_size=0.80)
print('Training data shape: {}'.format(X_train.shape))
print('Test data shape:     {}'.format(X_test.shape))

gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(X_train, y_train)

predictions = gbtree.predict(X_test)

train_data = transformer.invert(X_train)
test_data = transformer.invert(X_test)

preprocess = lambda z: transformer.transform(z)

# Initialize a TabularExplainer
explainers = TabularExplainer(
    explainers=["shap"],
    mode="classification",
    data=train_data,
    model=gbtree,
    preprocess=preprocess,
    params={
        "shap": {"nsamples": 100},
    }
)

def shap_explanation(instance):
    local_explanations = explainers.explain(X=instance)
    return local_explanations["shap"].plotly_plot(index=0, class_names=class_names).to_html()

