import gradio as gr

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
    # You can replace the following line with your server-side logic for credit card approval
    # For demonstration purposes, I'm returning a placeholder result.
    result = "Your credit card is approved!" if AMT_INCOME_TOTAL > 50000 else "Your credit card is not approved."

    return result
iface = gr.Interface(
    fn=credit_card_approval,
    inputs=[
        gr.Radio(['M', 'F'], label='Gender'),
        gr.Radio(['Yes', 'No'], label='Own car'),
        gr.Radio(['Yes', 'No'], label='Own realty'),
        gr.Number(0, label='Children count'),
        gr.Number(0, label='Total income'),
        gr.Dropdown(['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed', 'Student'],
                    label='Income type'),
        gr.Dropdown(['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary',
                     'Academic degree'],
                    label='Education type'),
        gr.Dropdown(['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
                    label='Family status'),
        gr.Dropdown(['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment',
                     'Co-op apartment'],
                    label='Housing type'),
        gr.Radio(['Yes', 'No'], label='Mobile'),
        gr.Radio(['Yes', 'No'], label='Work phone'),
        gr.Radio(['Yes', 'No'], label='Phone'),
        gr.Radio(['Yes', 'No'], label='Email'),
        gr.Dropdown(OCCUPATION_TYPE,
                    label='Occupation type'),
        gr.Number(0, label='Family members count'),
        gr.Number(0, label='Month balance')
    ],
    outputs=gr.Textbox(),
    live=True
)

iface.launch()