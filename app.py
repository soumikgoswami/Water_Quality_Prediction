import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Loads the dataset, handles missing values, converts date,
    creates time-based features and lagged features.
    """
    try:
        df = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as the app.")
        return None
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df = df.sort_values(by=['id', 'date']).copy()
    for col in ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    pollutants_for_lags = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
    for pollutant in pollutants_for_lags:
        df[f'{pollutant}_lag1'] = df.groupby('id')[pollutant].shift(1)
        df[f'{pollutant}_lag2'] = df.groupby('id')[pollutant].shift(2)
        df[f'{pollutant}_lag1'] = df.groupby('id')[f'{pollutant}_lag1'].fillna(method='bfill')
        df[f'{pollutant}_lag2'] = df.groupby('id')[f'{pollutant}_lag2'].fillna(method='bfill')
    df = df.drop(['date', 'NH4', 'BSK5', 'Suspended'], axis=1) # Drop original date and other non-target pollutants
    return df
@st.cache_resource
def train_and_save_model(df):
    """
    Trains the MultiOutputRegressor model and saves it along with feature columns.
    """
    pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
    feature_columns = [col for col in df.columns if col not in pollutants and col != 'id']
    X = df[feature_columns + ['id']] # Include 'id' for one-hot encoding
    y = df[pollutants]
    X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_encoded, y)
    joblib.dump(model, 'pollution_model.pkl')
    joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")
    st.success("Model and column structure saved successfully!")
    return model, X_encoded.columns.tolist()
def load_model_and_columns():
    """Loads the pre-trained model and feature columns."""
    if os.path.exists('pollution_model.pkl') and os.path.exists('model_columns.pkl'):
        model = joblib.load('pollution_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    return None, None
def main():
    st.set_page_config(page_title="Water Quality Prediction", layout="centered")
    st.title("💧 Water Quality Prediction App")
    st.markdown("""
    This application predicts water pollutant levels (O2, NO3, NO2, SO4, PO4, CL)
    for a given station ID and year.
    """)
    data_file_path = 'PB_All_2000_2021.csv'
    if not os.path.exists(data_file_path):
        st.error(f"Required data file '{data_file_path}' not found. Please upload it to the same directory as the app.")
        st.stop()
    with st.spinner("Loading and preprocessing data..."):
        df_processed = load_and_preprocess_data(data_file_path)
    if df_processed is None:
        st.stop()
    model, model_columns = load_model_and_columns()
    if model is None or model_columns is None:
        st.warning("Model files not found. Training a new model. This may take a moment...")
        model, model_columns = train_and_save_model(df_processed)
        if model is None or model_columns is None:
            st.error("Failed to train or load the model. Please check the data and dependencies.")
            st.stop()
    st.sidebar.header("Input Parameters")
    unique_station_ids = sorted(df_processed['id'].unique().tolist())
    station_id_input = st.sidebar.selectbox("Select Station ID", unique_station_ids)
    year_input = st.sidebar.slider("Select Year for Prediction", min_value=2022, max_value=2030, value=2024)
    month_input = st.sidebar.slider("Select Month for Prediction", min_value=1, max_value=12, value=7)
    if st.sidebar.button("Predict Water Quality"):
        input_data = pd.DataFrame({
            'year': [year_input],
            'month': [month_input],
            'month_sin': [np.sin(2 * np.pi * month_input / 12)],
            'month_cos': [np.cos(2 * np.pi * month_input / 12)],
            'id': [station_id_input]
        })
        last_data_for_station = df_processed[df_processed['id'] == station_id_input].tail(2)
        pollutants_for_lags = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        for pollutant in pollutants_for_lags:
            input_data[f'{pollutant}_lag1'] = last_data_for_station[pollutant].iloc[-1] if len(last_data_for_station) >= 1 else df_processed[pollutant].mean()
            input_data[f'{pollutant}_lag2'] = last_data_for_station[pollutant].iloc[-2] if len(last_data_for_station) >= 2 else df_processed[pollutant].mean()
        input_encoded = pd.get_dummies(input_data, columns=['id'])
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_columns] 
        try:
            predicted_pollutants = model.predict(input_encoded)[0]
            pollutant_names = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

            st.subheader(f"Predicted Water Quality for Station '{station_id_input}' in {year_input}-{month_input:02d}:")
            prediction_df = pd.DataFrame({
                "Pollutant": pollutant_names,
                "Predicted Level": predicted_pollutants
            })
            st.table(prediction_df.style.format({"Predicted Level": "{:.2f}"}))
            st.markdown("---")
            st.subheader("Interpretation of Results:")
            st.info("""
            These predictions are based on the trained Random Forest Regressor model.
            The values represent the estimated levels of each pollutant.
            """)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure that the 'PB_All_2000_2021.csv' file is correctly formatted and present.")

if __name__ == "__main__":
    main()
