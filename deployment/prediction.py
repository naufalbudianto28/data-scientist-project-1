import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load Best Model
with open ('best_model_xgb.pkl', 'rb') as xgb_file:
    model_xgb = pickle.load(xgb_file)

def run():
    st.title('Klasifikasi XGBoost pada Telemarketing Deposito')

    with st.form('telemarketing_deposit'):
        st.write('### Masukkan Data Klien')

        # Job
        job_options = ['bluecollar', 'management', 'technician', 'admin', 'services', 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student']
        job = st.selectbox('Pekerjaan', job_options)

        # Marital status
        marital_status_options = ['single', 'divorced', 'married']
        marital_status = st.radio('Status Pernikahan', marital_status_options)

        # Education
        education_options = ['primary', 'secondary', 'tertiary']
        education = st.radio('Pendidikan', education_options)

        # Balance
        balance = st.number_input('Saldo Rata-rata (€)', min_value=0)

        # Housing loan
        housing_loan_options = ['yes', 'no']
        housing_loan = st.checkbox('Punya Pinjaman Rumah?', housing_loan_options)

        # Personal loan
        personal_loan_options = ['yes', 'no']
        personal_loan = st.checkbox('Punya Pinjaman Pribadi?', personal_loan_options)

        # Contact method
        contact_options = ['cellular', 'telephone', 'unknown']
        contact = st.selectbox('Metode Kontak', contact_options)

        # Duration
        duration = st.slider('Durasi Kontak Terakhir (detik)', 0, 3600, 1800)

        # Campaign
        campaign = st.slider('Jumlah Upaya Kontak Selama Campaign', 1, 30, 10)

        # Pdays
        pdays = st.slider('Jumlah Hari Terakhir Dihubungi (Jika -1, belum pernah)', -1, 20, 10)

        # Previous
        previous = st.slider('Jumlah Upaya Kontak Sebelum Campaign', 0, 30, 5)

        # Poutcome
        poutcome_options = ['success', 'failure', 'unknown', 'other']
        poutcome = st.selectbox('Hasil Campaign Sebelumnya', poutcome_options)

        submit_button = st.form_submit_button('Prediksi')

    housing_loan = 'yes' if housing_loan else 'no'
    personal_loan = 'yes' if personal_loan else 'no'

    input_data = pd.DataFrame({
        'job': [job],
        'marital': [marital_status],
        'education': [education],
        'balance': [balance],
        'housing': [housing_loan],
        'loan': [personal_loan],
        'contact': [contact],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    st.dataframe(input_data)

    if submit_button:
        num_features = input_data[['balance', 'duration', 'campaign', 'pdays', 'previous']]
        cat_features = input_data[['job', 'marital', 'contact', 'poutcome', 'housing', 'loan']]
        cat_ordinal_feature = input_data[['education']]

        y_pred = model_xgb.predict(input_data)
        st.write('Klasifikasi: ', y_pred)

if __name__ == '__main__':
    run()