import streamlit as st
import requests
import helper as help
import pandas as pd
import json

tab1, tab2 = st.tabs(["Form", "Upload"])

def convert(list):
    return pd.Series(tuple(list), name='A').unique()

with tab1:
    age = st.number_input('Age', step=1)
    job = st.selectbox(
        'Job',
        convert(help.train()['job'].to_numpy()))
    marital = st.selectbox(
        'Marital',
        convert(help.train()['marital'].to_numpy()))
    education = st.selectbox(
        'Education',
        convert(help.train()['education'].to_numpy()))
    balance = st.number_input('Balance', step=1)
    housing = st.selectbox(
        'Housing',
        convert(help.train()['housing'].to_numpy()))
    loan = st.selectbox(
        'Loan',
        convert(help.train()['loan'].to_numpy()))
    defaults = st.selectbox(
        'Default',
        convert(help.train()['default'].to_numpy()))
    campaign = st.number_input('Campaign', step=1)

    if st.button('Process'):
        input = {
            "signature_name": "serving_default", 
            "instances":[
                {
                    "age": [age],
                    "job": [job],
                    "marital": [marital],
                    "education": [education],
                    "balance": [balance],
                    "housing": [housing],
                    "loan": [loan],
                    "default": [defaults],
                    "campaign": [campaign]
                }
            ]
        }

        data = json.dumps(input)
        headers = {"content-type": "application/json"}

        json_response = requests.post('http://ts-gigih.herokuapp.com/v1/models/model:predict', data=data)
        res = json_response.json()
        st.text(f"Probabilitas untuk berlangganan deposito berjangka adalah {round(100 * res['predictions'][0][0], 2)} %"
                f"untuk setuju")

with tab2:
    st.text('Under Constructions')
    # uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     dataframe = pd.read_csv(uploaded_file, delimiter=";")

