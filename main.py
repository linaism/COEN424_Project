import os
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import Flask, request, render_template
from omnixai.data.tabular import Tabular

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


def create_app():
    # preload_model()

    # create and configure the app
    app = Flask(__name__)

    # enable CORS
    CORS(app, resources={r'/*': {'origins': '*'}})

    # a simple page that says hello
    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            # Form was submitted, process the data
            gender = request.form.get('gender')
            ownCar = request.form.get('ownCar')
            ownRealty = request.form.get('ownRealty')
            childrenCount = request.form.get('childrenCount')
            totalIncome = request.form.get('totalIncome')
            incomeType = request.form.get('incomeType')
            educationType = request.form.get('educationType')
            familyStatus = request.form.get('familyStatus')
            housingType = request.form.get('housingType')
            occupationType = request.form.get('occupationType')
            mobile = request.form.get('mobile')
            workPhone = request.form.get('workPhone')
            phone = request.form.get('phone')
            email = request.form.get('email')
            familyMembersCount = request.form.get('familyMembersCount')
            monthBalance = request.form.get('monthBalance')

            # Process the form data as needed

            # Pass the result to the template
            result = credit_card_approval(gender, 
                                          ownCar, 
                                          ownRealty, 
                                          childrenCount, 
                                          totalIncome, 
                                          incomeType, 
                                          educationType, 
                                          familyStatus,
                                          housingType, 
                                          occupationType, 
                                          mobile, 
                                          workPhone, 
                                          phone, 
                                          email, 
                                          familyMembersCount, 
                                          monthBalance)
            return render_template('index.html', result=result, 
                                   NAME_INCOME_TYPE=NAME_INCOME_TYPE,
                           NAME_EDUCATION_TYPE=NAME_EDUCATION_TYPE,
                           NAME_FAMILY_STATUS=NAME_FAMILY_STATUS,
                           NAME_HOUSING_TYPE=NAME_HOUSING_TYPE,
                           OCCUPATION_TYPE=OCCUPATION_TYPE)

        # If it's a GET request or the form hasn't been submitted yet, render the form template
        return render_template('index.html', result=None,
                            NAME_INCOME_TYPE=NAME_INCOME_TYPE,
                           NAME_EDUCATION_TYPE=NAME_EDUCATION_TYPE,
                           NAME_FAMILY_STATUS=NAME_FAMILY_STATUS,
                           NAME_HOUSING_TYPE=NAME_HOUSING_TYPE,
                           OCCUPATION_TYPE=OCCUPATION_TYPE)

    return app


app = create_app()

if __name__ == "__main__":
    #    app = create_app()
    app.run(host="0.0.0.0", port=5000)