import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Import ulang semua komponen yang digunakan di pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Fungsi untuk load model
@st.cache_resource
def load_model():
    model = joblib.load("best_model_RandomForest.pkl")
    return model

# Load model
model = load_model()

# Judul halaman
st.set_page_config(page_title="Prediksi Harga Mobil", layout="centered")
st.title("🚗 Prediksi Harga Mobil Bekas")

st.markdown("""
Masukkan detail mobil di bawah ini untuk memprediksi harga jualnya.
""")

# Input user
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Tahun Mobil (Year)", min_value=1990, max_value=2025, value=2015)
    present_price = st.number_input("Harga Baru Mobil (dalam lakhs)", min_value=0.0, value=5.0, step=0.1)
    kms_driven = st.number_input("Jarak Tempuh (KMs Driven)", min_value=0, value=30000, step=500)
    owner = st.selectbox("Jumlah Pemilik Sebelumnya (Owner)", [0, 1, 3])

with col2:
    fuel_type = st.selectbox("Jenis Bahan Bakar (Fuel Type)", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Tipe Penjual (Seller Type)", ["Dealer", "Individual", "Trustmark Dealer"])
    transmission = st.selectbox("Transmisi (Transmission)", ["Manual", "Automatic"])

# Buat DataFrame dari input
input_data = pd.DataFrame({
    "Year": [year],
    "Present_Price": [present_price],
    "Kms_Driven": [kms_driven],
    "Owner": [owner],
    "Fuel_Type": [fuel_type],
    "Seller_Type": [seller_type],
    "Transmission": [transmission]
})

# Tampilkan input
st.subheader("Data yang Dimasukkan")
st.dataframe(input_data)

# Prediksi
if st.button("Prediksi Harga Jual"):
    try:
        log_pred = model.predict(input_data)
        final_pred = np.expm1(log_pred)[0]  # Target sebelumnya dilog-transformed (log1p)
        st.success(f"💰 Perkiraan harga jual mobil: **Rp {final_pred:,.0f}**")
    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi.")
        st.exception(e)
