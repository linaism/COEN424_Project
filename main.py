import gradio as gr
import joblib
import numpy as np
import pandas as pd

from omnixai.data.tabular import Tabular

# Load the pre-trained machine learning model 
model = joblib.load('model.pkl')
transformer = joblib.load('transformer.pkl')

COLUMNS =['CODE_GENDER', 
          'FLAG_OWN_CAR', 
          'FLAG_OWN_REALTY', 
          'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 
       'NAME_INCOME_TYPE', 
       'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 
       'NAME_HOUSING_TYPE', 
       'FLAG_MOBIL',
       'FLAG_WORK_PHONE', 
       'FLAG_PHONE', 
       'FLAG_EMAIL', 
       'OCCUPATION_TYPE',
       'CNT_FAM_MEMBERS', 
       'MONTHS_BALANCE']

CATEGORICAL_FEATURES = [
    "NAME_INCOME_TYPE", 
    "NAME_EDUCATION_TYPE", 
    "NAME_FAMILY_STATUS", 
    "NAME_HOUSING_TYPE", 
    "OCCUPATION_TYPE"
    ]

NAME_INCOME_TYPE = ['Working', 
                    'Commercial associate',
                    'Pensioner',
                    'State servant',
                    'Student']
NAME_EDUCATION_TYPE = ['Higher education',
                    'Secondary / secondary special', 
                    'Incomplete higher',
                    'Lower secondary', 
                    'Academic degree']

NAME_FAMILY_STATUS = ['Civil marriage', 
                      'Married', 
                      'Single / not married', 
                      'Separated', 
                      'Widow']

NAME_HOUSING_TYPE = ['Rented apartment', 
                     'House / apartment', 
                     'Municipal apartment', 
                     'With parents', 
                     'Co-op apartment', 
                     'Office apartment']

OCCUPATION_TYPE = ['Security staff', 
                   'Sales staff', 
                   'Accountants', 
                   'Laborers', 
                   'Managers', 
                   'Drivers', 
                   'Core staff', 
                   'High skill tech staff', 
                   'Cleaning staff', 
                   'Private service staff', 
                   'Cooking staff', 
                   'Low-skill Laborers', 
                   'Medicine staff', 
                   'Secretaries', 
                   'Waiters/barmen staff', 
                   'HR staff', 
                   'Realty agents', 
                   'IT staff']

def credit_card_approval(CODE_GENDER,
    FLAG_OWN_CAR,
    FLAG_OWN_REALTY,
    CNT_CHILDREN,
    AMT_INCOME_TOTAL,
    NAME_INCOME_TYPE,
    NAME_EDUCATION_TYPE,
    NAME_FAMILY_STATUS,
    NAME_HOUSING_TYPE,
    FLAG_MOBIL,
    FLAG_WORK_PHONE,
    FLAG_PHONE,
    FLAG_EMAIL,
    OCCUPATION_TYPE,
    CNT_FAM_MEMBERS,
    MONTHS_BALANCE
):
    try:

        # Perform any necessary data preprocessing
        input_data = np.array([[CODE_GENDER,
            FLAG_OWN_CAR,
            FLAG_OWN_REALTY,
            CNT_CHILDREN,
            AMT_INCOME_TOTAL,
            NAME_INCOME_TYPE,
            NAME_EDUCATION_TYPE,
            NAME_FAMILY_STATUS,
            NAME_HOUSING_TYPE,
            FLAG_MOBIL,
            FLAG_WORK_PHONE,
            FLAG_PHONE,
            FLAG_EMAIL,
            OCCUPATION_TYPE,
            CNT_FAM_MEMBERS,
            MONTHS_BALANCE]])


        data = pd.DataFrame(input_data, columns=COLUMNS)
        print(data)

        gender_flag = {'M' : 1,
                       'F' : 0}
        binary_flag = {'Yes' : 1,
                       'No': 0}
        data["CODE_GENDER"] = data['CODE_GENDER'].map(gender_flag)

        columns = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL", "FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL"]
        for col in columns:
            data[col] = data[col].map(binary_flag)

        print(data)

        # Make a prediction using the loaded model
        # prediction = model.predict(input_data)
        # Preprocess the input
        table = Tabular(
            data=data,
            categorical_columns=CATEGORICAL_FEATURES,
        )
        transformed_input = transformer.transform(table)

        # Make a prediction using the loaded model
        prediction = model.predict(transformed_input)

        print(prediction)
        return prediction[0]
    except Exception as e:
        return str(e)

    
iface = gr.Interface(
    fn=credit_card_approval,
    inputs=[
        gr.Radio(['M', 'F'], value="M", label='Gender'),
        gr.Radio(['Yes', 'No'], value="Yes", label='Own car'),
        gr.Radio(['Yes', 'No'], value="Yes", label='Own realty'),
        gr.Number(0, label='Children count'),
        gr.Number(0, label='Total income'),
        gr.Dropdown(NAME_INCOME_TYPE,
                    value="Working",
                    label='Income type'),
        gr.Dropdown(NAME_EDUCATION_TYPE,
                     value="Higher education",
                    label='Education type'),
        gr.Dropdown(NAME_FAMILY_STATUS,
                    value="Single / not married",
                    label='Family status'),
        gr.Dropdown(NAME_HOUSING_TYPE,
                     value="Rented apartment",
                    label='Housing type'),
        gr.Radio(['Yes', 'No'], value="Yes", label='Mobile'),
        gr.Radio(['Yes', 'No'], value="Yes", label='Work phone'),
        gr.Radio(['Yes', 'No'], value="Yes", label='Phone'),
        gr.Radio(['Yes', 'No'], value="Yes", label='Email'),
        gr.Dropdown(OCCUPATION_TYPE,
                    value="HR staff",
                    label='Occupation type'),
        gr.Number(0, label='Family members count'),
        gr.Number(0, label='Month balance')
    ],
    outputs="number",
    live=False
)



# Define the predict function for the Gradio interface
def predict(input_data):
    try:
        prediction = credit_card_approval(np.array(input_data))
        # Perform any necessary data preprocessing
        input_data = np.array(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(input_data.reshape(1, -1))

        return prediction.tolist()
    except Exception as e:
        return str(e)

iface.launch()
