import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as stc

# ========== Memuat Model ========== #
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

# ========== Tampilan Utama ========== #
html_temp = """
<div style="background-color:#000;padding:10px;border-radius:10px">
    <h1 style="color:#fff;text-align:center">🚗 Aplikasi Prediksi Harga Mobil</h1> 
    <h4 style="color:#fff;text-align:center">Dibangun dengan Model Random Forest</h4> 
</div>
"""

desc_temp = """
### Tentang Aplikasi Ini  
Aplikasi ini memungkinkan pengguna untuk memasukkan berbagai spesifikasi mobil dan langsung menerima prediksi harga mobil berdasarkan model yang telah dilatih.

#### Sumber Data
Kaggle: Car Price Prediction Dataset  
Model: Random Forest Regressor yang sudah dituning
"""

# ========== Fungsi Utama ========== #
def main():
    stc.html(html_temp)
    menu = ["Beranda", "Prediksi Harga"]
    choice = st.sidebar.selectbox("📋 Menu", menu)

    if choice == "Beranda":
        st.subheader("🏠 Beranda")
        st.markdown(desc_temp, unsafe_allow_html=True)

    elif choice == "Prediksi Harga":
        jalankan_aplikasi_prediksi()

# ========== Aplikasi Prediksi ========== #
def jalankan_aplikasi_prediksi():
    st.subheader("Masukkan Spesifikasi Mobil")

    with st.form("form_prediksi"):
        prod_year = st.number_input("Tahun Produksi", min_value=1990, max_value=2025, value=2015)
        engine_volume = st.number_input("Volume Mesin (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
        mileage = st.number_input("Jarak Tempuh (km)", min_value=0, max_value=1_000_000, step=1000, value=150_000)
        levy = st.number_input("Levy", min_value=0, max_value=50000, step=100, value=0)
        cylinders = st.number_input("Jumlah Silinder", min_value=1, max_value=16, value=4)

        manufacturer = st.selectbox("Merek", ['Toyota', 'BMW', 'Mercedes-Benz', 'Hyundai', 'Rare'])
        fuel_type = st.selectbox("Tipe Bahan Bakar", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
        gearbox = st.selectbox("Transmisi", ['Automatic', 'Manual'])
        drive_wheels = st.selectbox("Penggerak Roda", ['front', 'rear', '4x4'])
        leather = st.selectbox("Interior Kulit", ['Yes', 'No'])
        right_hand = st.selectbox("Setir Kanan", ['Yes', 'No'])
        doors = st.number_input("Jumlah Pintu", min_value=2, max_value=6, value=4)

        submitted = st.form_submit_button("🔍 Prediksi Harga")

    if submitted:
        try:
            # Rekayasa Fitur
            volume_per_cylinder = engine_volume / cylinders
            fuel_gear = fuel_type + "_" + gearbox
            car_age = 2025 - prod_year

            def kategorikan_pintu(jumlah_pintu):
                if jumlah_pintu <= 3:
                    return '2-3'
                elif jumlah_pintu <= 5:
                    return '4-5'
                else:
                    return '>5'

            doors_category = kategorikan_pintu(doors)
            leather_bin = 1 if leather == 'Yes' else 0
            right_hand_bin = 1 if right_hand == 'Yes' else 0

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

            # Prediksi
            prediction = model.predict(input_df)[0]
            st.success(f"💰 Prediksi Harga Mobil: **${prediction:,.2f}**")

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# ========== Menjalankan Aplikasi ========== #
if __name__ == '__main__':
    main()
