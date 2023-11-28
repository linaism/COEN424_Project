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

# Confirm that binary categorical variables have been converted to numerical 
print(application_record.head())

# Drop unnessessary columns
application_record.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True, axis=1)

# Replace space with dash
# application_record['OCCUPATION_TYPE'] = application_record['OCCUPATION_TYPE'].str.replace(' ', '-')

# Confirm that columns have been dropped
print(application_record.head())

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

# Confirm the above
print(credit_record['STATUS'].value_counts())

# Merge both credit and applications records to create a comprehensive dataset
df_credit = application_record.merge(credit_record, how='inner', on=['ID'])
print(df_credit.head())
df_credit.drop(['ID'], inplace=True, axis=1)

print(df_credit['CODE_GENDER'].dtype) # int
print(df_credit['FLAG_OWN_CAR'].dtype) #int
print(df_credit['FLAG_OWN_REALTY'].dtype) #int
print(df_credit['AMT_INCOME_TOTAL'].dtype) #float
print(df_credit['NAME_INCOME_TYPE'].dtype) #string
print(df_credit['NAME_EDUCATION_TYPE'].dtype) #string
print(df_credit['NAME_FAMILY_STATUS'].dtype) #string
print(df_credit['NAME_HOUSING_TYPE'].dtype) #string
print(df_credit['FLAG_MOBIL'].dtype) #int
print(df_credit['FLAG_WORK_PHONE'].dtype) #int
print(df_credit['FLAG_PHONE'].dtype) #int
print(df_credit['FLAG_EMAIL'].dtype) #int
print(df_credit['OCCUPATION_TYPE'].dtype) #string
print(df_credit['CNT_FAM_MEMBERS'].dtype) #float
print(df_credit['MONTHS_BALANCE'].dtype) #int

print(df_credit['NAME_INCOME_TYPE'].unique())
print(df_credit['NAME_EDUCATION_TYPE'].unique())
print(df_credit['NAME_FAMILY_STATUS'].unique())
print(df_credit['NAME_HOUSING_TYPE'].unique())

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
print(tabular_data)

# Train an XGBoost model
np.random.seed(1)
transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
x = transformer.transform(tabular_data)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x[:, :-1], x[:, -1], train_size=0.80)
# X_train.to_csv('train_data.csv', index=False)
np.savetxt('train_data.csv', X_train, delimiter=',')
print('Training data shape: {}'.format(X_train.shape))
print('Test data shape:     {}'.format(X_test.shape))

gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(X_train, y_train)

joblib.dump(gbtree, 'model.pkl')
joblib.dump(transformer, 'transformer.pkl')

predictions = gbtree.predict(X_test)
print('Test accuracy: {}'.format(
    sklearn.metrics.accuracy_score(y_test, predictions)))
print(predictions)

test_x_data = transformer.invert(X_test).to_numpy()

indices = np.where(predictions == 0)
for i in indices[0]: 
    print("Index", i)
    print(test_x_data[i])

# print("TESTTTTTTTT")

# test_instance = pd.DataFrame(np.array([[1, 0, 0, 0, 40500.0, 'Working', "Higher education", "Married", "Rented apartment",1,0,0,0,"Laborers",3.0,-21]]), 
#                    columns=['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
#        'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
#        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL',
#        'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE',
#        'CNT_FAM_MEMBERS', 'MONTHS_BALANCE'])

# print(test_instance)

# input_transformed = transformer.transform(test_instance)
# print(input_transformed)
# print(input_transformed.shape())

# print("TEST")
# print(gbtree.predict(transformer.transform(test_instance)))

# print(transformer.invert(np.array([X_test[0],])))
# print(transformer.transform(np.array([X_test[0],])))

# print(gbtree.predict(np.array([X_test[0],])))

# Convert the transformed data back to Tabular instances
train_data = transformer.invert(X_train)
test_data = transformer.invert(X_test)

# print('data at 0')
# print(test_data[0])
# print(transformer.transform(test_data[0]))

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

# # Generate explanations
test_instances = test_data[1653:1658]
local_explanations = explainers.explain(X=test_instances)
# global_explanations = explainers.explain_global(
#     params={"pdp": {"features": ["CODE_GENDER", "FLAG_OWN_CAR", 'FLAG_OWN_REALTY', 
#               'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
#               'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 
#               'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'MONTHS_BALANCE', ]}}
# )

# print("SHAP results:")
# local_explanations["shap"].ipython_plot(index=1, class_names=class_names)
# local_explanations["shap"].plot(index=1, class_names=class_names, max_num_subplots=4).show()
# print(local_explanations["shap"].plotly_plot(index=1, class_names=class_names))

# predict_function=lambda z: gbtree.predict_proba(transformer.transform(z))

# explainer = GPTExplainer(
#     training_data=tabular_data,
#     predict_function=predict_function,
#     apikey="sk-xxx"
# )
# # Apply an inverse transform, i.e., converting the numpy array back to `Tabular`
# test_instance = transformer.invert(X_test)
# test_x = test_instance[1654]

# print(test_x.data.iloc[0])
# print(test_x.categorical_cols)

# explanations = explainer.explain(test_x)
# print(explanations.get_explanations(index=1)["text"])