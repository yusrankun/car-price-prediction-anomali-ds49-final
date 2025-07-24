import streamlit as st
import pandas as pd
import pickle

# ========== Load Model ==========
@st.cache_resource
def load_model():
    with open("best_model_RandomForest.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ========== Homepage ==========
def main():
    st.title("🚗 Car Price Prediction App")
    st.markdown("Prediksi harga mobil berdasarkan spesifikasi yang Anda masukkan.")
    
    menu = ["Home", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini menggunakan model **Random Forest** untuk memprediksi harga mobil berdasarkan berbagai fitur seperti tahun produksi, kapasitas mesin, jenis bahan bakar, transmisi, dan lainnya.  
        Dataset berasal dari: **Car Price Prediction Dataset (Kaggle)**.
        """)
    else:
        run_prediction()

# ========== Preprocessing Function ==========
def preprocess_input(data):
    # Feature Engineering
    data['volume_per_cylinder'] = data['Engine volume'] / data['Cylinders']
    data['fuel_gear'] = data['Fuel type'] + "_" + data['Gear box type']
    data['car_age'] = 2025 - data['Prod. year']

    # Doors Category
    def categorize_doors(val):
        if val <= 3:
            return '2-3'
        elif val <= 5:
            return '4-5'
        else:
            return '>5'
    
    data['Doors_category'] = data['Doors'].apply(categorize_doors)

    # Binary Encoding
    data['Leather interior'] = data['Leather interior'].map({'Yes': 1, 'No': 0})
    data['Right_hand_drive'] = data['Right_hand_drive'].map({'Yes': 1, 'No': 0})

    # Final selected features (same as Colab)
    selected_features = ['Prod. year', 'Engine volume', 'Mileage', 'Levy', 'Cylinders',
                         'volume_per_cylinder', 'Fuel type', 'Gear box type',
                         'Drive wheels', 'Leather interior', 'Right_hand_drive',
                         'Manufacturer', 'fuel_gear', 'car_age', 'Doors_category']
    
    return data[selected_features]

# ========== Prediction Page ==========
def run_prediction():
    st.subheader("🧠 Masukkan Spesifikasi Mobil Anda")

    with st.form("input_form"):
        prod_year = st.number_input("Tahun Produksi", 1990, 2025, 2015)
        engine_volume = st.number_input("Volume Mesin (L)", 0.5, 10.0, step=0.1, value=2.0)
        mileage = st.number_input("Jarak Tempuh (km)", 0, 1_000_000, step=1000, value=150_000)
        levy = st.number_input("Levy", 0, 50000, step=100, value=0)
        cylinders = st.number_input("Jumlah Silinder", 1, 16, value=4)
        doors = st.number_input("Jumlah Pintu", 2, 6, value=4)

        manufacturer = st.selectbox("Merek", ['Toyota', 'BMW', 'Mercedes-Benz', 'Hyundai', 'Rare'])
        fuel_type = st.selectbox("Tipe Bahan Bakar", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
        gearbox = st.selectbox("Transmisi", ['Automatic', 'Manual'])
        drive_wheels = st.selectbox("Penggerak Roda", ['front', 'rear', '4x4'])
        leather = st.selectbox("Interior Kulit", ['Yes', 'No'])
        right_hand = st.selectbox("Setir Kanan", ['Yes', 'No'])

        submitted = st.form_submit_button("🔍 Prediksi Harga")

    if submitted:
        try:
            input_dict = {
                'Prod. year': [prod_year],
                'Engine volume': [engine_volume],
                'Mileage': [mileage],
                'Levy': [levy],
                'Cylinders': [cylinders],
                'Fuel type': [fuel_type],
                'Gear box type': [gearbox],
                'Drive wheels': [drive_wheels],
                'Leather interior': [leather],
                'Right_hand_drive': [right_hand],
                'Manufacturer': [manufacturer],
                'Doors': [doors]
            }

            input_df = pd.DataFrame(input_dict)
            processed_df = preprocess_input(input_df)

            pred = model.predict(processed_df)[0]
            st.success(f"💰 Estimasi Harga Mobil Anda: **${pred:,.2f}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ========== Run App ==========
if __name__ == "__main__":
    main()
