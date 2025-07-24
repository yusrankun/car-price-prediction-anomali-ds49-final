# app.py

import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import pickle

# ========== Load Model ==========
@st.cache_resource
def load_model():
    with open("best_model_RandomForest.pkl", "rb") as f:
        return pickle.load(f)

# ========== HTML Template ==========
html_temp = """<div style="background-color:#1f77b4;padding:10px;border-radius:10px">
                <h1 style="color:white;text-align:center">🚗 Car Price Prediction App</h1> 
                <h4 style="color:white;text-align:center">Made for: Final Project</h4> 
                </div>
            """

desc_temp = """
### About this App
This app uses a machine learning model (Random Forest) to predict used car prices based on their specifications.

#### Key Features:
- Clean UI with dynamic input fields.
- Automatic preprocessing & feature engineering.
- Real-time prediction using your trained model.

#### Model & Data
- **Model**: Random Forest Regressor
- **Data**: Used Car Dataset
"""

# ========== Doors Helper Function ==========
def categorize_doors(door_value):
    try:
        door = float(door_value)
        if door <= 3:
            return '2-3'
        elif door <= 5:
            return '4-5'
        else:
            return '>5'
    except:
        return 'Unknown'

# ========== ML App ==========
def run_ml_app():
    st.subheader("🔍 Car Price Prediction")

    with st.form("form_input"):
        prod_year = st.number_input("Tahun Produksi", min_value=1990, max_value=2025, value=2015)
        engine_volume = st.number_input("Volume Mesin (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
        cylinders = st.number_input("Jumlah Silinder", min_value=1, max_value=16, value=4)
        mileage = st.number_input("Jarak Tempuh (km)", min_value=0, max_value=1_000_000, step=1000, value=150000)
        levy = st.number_input("Levy", min_value=0, max_value=50000, step=100, value=0)
        doors = st.number_input("Jumlah Pintu", min_value=2, max_value=8, value=4)

        leather_interior = st.selectbox("Interior Kulit", ['Yes', 'No'])
        wheel_position = st.selectbox("Posisi Stir", ['Left-hand drive', 'Right-hand drive'])

        manufacturer = st.selectbox("Merek", ['Toyota', 'BMW', 'Mercedes-Benz', 'Hyundai', 'Rare'])
        fuel_type = st.selectbox("Tipe Bahan Bakar", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
        gearbox = st.selectbox("Transmisi", ['Automatic', 'Manual'])
        drive_wheels = st.selectbox("Penggerak Roda", ['front', 'rear', '4x4'])

        submit = st.form_submit_button("💰 Prediksi Harga")

    if submit:
        df = pd.DataFrame({
            'Prod. year': [prod_year],
            'Engine volume': [engine_volume],
            'Cylinders': [cylinders],
            'Mileage': [mileage],
            'Levy': [levy],
            'Doors': [doors],
            'Leather interior': [1 if leather_interior == 'Yes' else 0],
            'Right_hand_drive': [1 if 'Right' in wheel_position else 0],
            'Manufacturer': [manufacturer],
            'Fuel type': [fuel_type],
            'Gear box type': [gearbox],
            'Drive wheels': [drive_wheels],
        })

        # Feature Engineering
        df['volume_per_cylinder'] = df['Engine volume'] / df['Cylinders']
        df['fuel_gear'] = df['Fuel type'] + "_" + df['Gear box type']
        df['car_age'] = 2025 - df['Prod. year']
        df['Doors_category'] = df['Doors'].apply(categorize_doors)
        df.drop(columns=['Doors'], inplace=True)

        try:
            prediction = model.predict(df)[0]
            st.success(f"💲 Estimasi Harga Mobil: **${prediction:,.2f}**")
        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")

# ========== Main App Layout ==========
def main():
    stc.html(html_temp)
    menu = ["Home", "Car Price Prediction"]
    choice = st.sidebar.selectbox("📁 Menu", menu)

    if choice == "Home":
        st.subheader("🏠 Home")
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Car Price Prediction":
        run_ml_app()

if __name__ == "__main__":
    main()
