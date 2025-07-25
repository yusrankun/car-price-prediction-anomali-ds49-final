import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as stc

# ========== Load Model ========== #
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model_RandomForest.pkl")
        st.success("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

# ========== HTML Template ========== #
html_temp = """
<div style="padding:10px;border-radius:10px; background-color:rgba(0,0,0,0.8)">
    <h1 style="color:white;text-align:center;">🚗 Aplikasi Prediksi Harga Mobil</h1> 
    <h4 style="color:white;text-align:center;">Model: Random Forest Regressor</h4> 
</div>
"""

desc_temp = """
### Tentang Aplikasi  
Aplikasi ini memungkinkan Anda memasukkan spesifikasi mobil dan mendapatkan prediksi harga secara instan berdasarkan model yang telah dilatih.

#### Sumber Data  
Kaggle - *Car Price Prediction Dataset*  
Model yang digunakan: Random Forest (hasil tuning)
"""

# ========== Fungsi Utama ========== #
def main():
    stc.html(html_temp)
    menu = ["Beranda", "Prediksi Harga"]
    choice = st.sidebar.selectbox("📋 Menu Navigasi", menu)

    if choice == "Beranda":
        st.subheader("🏠 Beranda")
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Prediksi Harga":
        run_ml_app()

# ========== Fungsi Prediksi ========== #
def run_ml_app():
    st.subheader("🧠 Masukkan Spesifikasi Mobil")

    with st.form("prediction_form"):
        prod_year = st.number_input("Tahun Produksi", min_value=1990, max_value=2025, value=2015)
        engine_volume = st.number_input("Volume Mesin (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
        mileage = st.number_input("Jarak Tempuh (km)", min_value=0, max_value=1_000_000, step=1000, value=150_000)
        levy = st.number_input("Levy (Pajak)", min_value=0, max_value=50000, step=100, value=0)
        cylinders = st.number_input("Jumlah Silinder", min_value=1, max_value=16, value=4)

        manufacturer = st.selectbox("Merek Mobil", ['Toyota', 'BMW', 'Mercedes-Benz', 'Hyundai', 'Rare'])
        fuel_type = st.selectbox("Jenis Bahan Bakar", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
        gearbox = st.selectbox("Transmisi", ['Automatic', 'Manual'])
        drive_wheels = st.selectbox("Penggerak Roda", ['front', 'rear', '4x4'])
        leather = st.selectbox("Interior Kulit?", ['Ya', 'Tidak'])
        right_hand = st.selectbox("Setir Kanan?", ['Ya', 'Tidak'])
        doors = st.number_input("Jumlah Pintu", min_value=2, max_value=6, value=4)

        submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        try:
            # Rekayasa Fitur
            volume_per_cylinder = engine_volume / cylinders
            fuel_gear = fuel_type + "_" + gearbox
            car_age = 2025 - prod_year

            def categorize_doors(door_value):
                if door_value <= 3:
                    return '2-3'
                elif door_value <= 5:
                    return '4-5'
                else:
                    return '>5'

            doors_category = categorize_doors(doors)
            leather_bin = 1 if leather == 'Ya' else 0
            right_hand_bin = 1 if right_hand == 'Ya' else 0

            input_df = pd.DataFrame({
                'Prod. year': [prod_year],
                'Engine volume': [engine_volume],
                'Mileage': [mileage],
                'Levy': [levy],
                'Manufacturer': [manufacturer],
                'Fuel type': [fuel_type],
                'Gear box type': [gearbox],
                'Drive wheels': [drive_wheels],
                'Leather interior': [leather_bin],
                'Right_hand_drive': [right_hand_bin],
                'Cylinders': [cylinders],
                'volume_per_cylinder': [volume_per_cylinder],
                'fuel_gear': [fuel_gear],
                'car_age': [car_age],
                'Doors_category': [doors_category]
            })

            # Prediksi Harga
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Estimasi Harga Mobil: **${prediction:,.2f}**")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# ========== Jalankan Aplikasi ========== #
if __name__ == '__main__':
    main()
