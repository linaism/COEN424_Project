import os
import io
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from flask import Flask, request, render_template, render_template_string, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
from preprocess import explainers, transformer, class_names
from db import MongoDB
from doc_models.results_model import Results
from datetime import datetime
import matplotlib

matplotlib.use('agg')

table = None

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

        gender_flag = {'M' : 1,
                       'F' : 0}
        binary_flag = {'Yes' : 1,
                       'No': 0}
        prediction_flag = {0: 'Not approved',
                           1: 'Approved'}
        data["CODE_GENDER"] = data['CODE_GENDER'].map(gender_flag)

        columns = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL", "FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL"]
        for col in columns:
            data[col] = data[col].map(binary_flag)

        model = joblib.load('model.pkl')
        # transformer = joblib.load('transformer.pkl')

        # Make a prediction using the loaded model
        # prediction = model.predict(input_data)
        # Preprocess the input
        global table
        table = Tabular(
            data=data,
            categorical_columns=CATEGORICAL_FEATURES,
        )
        transformed_input = transformer.transform(table)

        # Make a prediction using the loaded model
        prediction = model.predict(transformed_input)

        pred_value = prediction_flag.get(prediction[0])

        docmodel = Results(TIMESTAMP=datetime.utcnow(),
            CODE_GENDER=CODE_GENDER,
            FLAG_OWN_CAR=FLAG_OWN_CAR,
            FLAG_OWN_REALTY=FLAG_OWN_REALTY,
            CNT_CHILDREN=CNT_CHILDREN,
            AMT_INCOME_TOTAL=AMT_INCOME_TOTAL,
            NAME_INCOME_TYPE=NAME_INCOME_TYPE,
            NAME_EDUCATION_TYPE=NAME_EDUCATION_TYPE,
            NAME_FAMILY_STATUS=NAME_FAMILY_STATUS,
            NAME_HOUSING_TYPE=NAME_HOUSING_TYPE,
            FLAG_MOBIL=FLAG_MOBIL,
            FLAG_WORK_PHONE=FLAG_WORK_PHONE,
            FLAG_PHONE=FLAG_PHONE,
            FLAG_EMAIL=FLAG_EMAIL,
            OCCUPATION_TYPE=OCCUPATION_TYPE,
            CNT_FAM_MEMBERS=CNT_FAM_MEMBERS,
            MONTHS_BALANCE=MONTHS_BALANCE,
            PREDICTION=pred_value)
        save_result = app.mongo.save_results(docmodel.to_json(),)

        # display_results = app.mongo.get_all_results()
        return pred_value
    except Exception as e:
        return str(e)
    

def dict_to_table(dict_list):
    rows = [] 
    for item in dict_list: 
        entry = []
        for key, value in item.items(): 
            entry.append(value)
        rows.append(entry)
    return rows


def create_app():
    # create and configure the app
    app = Flask(__name__)

    # enable CORS
    CORS(app, resources={r'/*': {'origins': '*'}})
    MONGO_URI = os.environ.get("MONGO_URI")

    with app.app_context():
        mongo = MongoDB(MONGO_URI)
        mongo.init_app(app)
        app.mongo = mongo

    @app.route('/plot.png')
    def plot_png():
        fig = create_figure()
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    def create_figure():
        local_explanations = explainers.explain(X=table)
        plt = local_explanations["shap"].plot(index=0, class_names=class_names, max_num_subplots=4)
        return plt
    
    @app.route('/history', methods=['GET'])
    def history():
        res = app.mongo.get_all_results()
        display_results = dict_to_table(list(res))
        return render_template('history.html', display_results=display_results)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        print("In index")
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
                                          mobile, 
                                          workPhone, 
                                          phone, 
                                          email, 
                                          occupationType, 
                                          familyMembersCount, 
                                          monthBalance)

            return render_template('index.html', result=result, shap_img="shap_explanation.png",
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