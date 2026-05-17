import streamlit as st
import pandas as pd
import pickle
import os

# 1. Az oldal alapbeállításai (Böngésző fül címe és ikonja)
st.set_page_config(
    page_title="Medical Insurance Predictor", 
    page_icon="🏥", 
    layout="centered"
)

# Címsor és leírás
st.title("🏥 Egészségbiztosítási Díjkalkulátor")
st.write("Ez az alkalmazás egy Scikit-Learn Pipeline és Random Forest Regressor segítségével, valós adatok alapján becsüli meg az éves biztosítási díjat.")

# 2. A modell betöltése
@st.cache_resource
def load_model():
    # Megkeressük a feltöltött pkl fájlt
    model_path = "insurance_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"Hiba: A '{model_path}' fájl nem található a GitHub repódban! Kérlek, töltsd fel az app.py mellé.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

pipeline = load_model()

# Csak akkor jelenítjük meg az űrlapot, ha a modell sikeresen betöltődött
if pipeline is not None:
    st.divider()
    st.subheader("Kérlek, add meg az ügyfél adatait:")

    # Az űrlap elemeit két esztétikus oszlopba rendezzük
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Életkor (év)", min_value=18, max_value=100, value=30, step=1)
        sex = st.selectbox("Nem", ["male", "female"])
        bmi = st.number_input("Testtömeg-index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Gyermekek száma", min_value=0, max_value=5, value=0, step=1)
    smoker = st.selectbox("Dohányzik?", ["no", "yes"])
    region = st.selectbox("Régió (USA)", ["southwest", "southeast", "northwest", "northeast"])

st.divider()

# 3. Predikció indítása a gomb megnyomására
if st.button("Biztosítási díj kiszámítása", type="primary"):
    # Fontos: Pontosan olyan oszlopnevekkel és formátumban adjuk át az adatot,
    # ahogy azt a tanítás során a DataFrame-ben megszokta a modell!
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])
    
    # Elvégezzük a jóslást a beágyazott pipeline-on keresztül
    try:
        predicted_charge = pipeline.predict(input_data)[0]
        
        # Eredmény látványos, színes dobozban való megjelenítése
        st.subheader("A modell által becsült éves költség:")
        st.success(f"💰 **${predicted_charge:,.2f}** / év")
        
        # Egy kis extra üzleti logika (Magyarázat a tanárnak a modell működéséről)
        if smoker == "yes":
            st.warning("⚠️ **Elemzési megjegyzés:** A modell kiemelten magas díjat számolt fel, mivel a dohányzás a legmeghatározóbb feature a regressziós fák döntési folyamatában.")
        elif bmi >= 30:
            st.info("💡 **Elemzési megjegyzés:** A 30 feletti BMI érték (elhízás kategória) a modelledben statisztikailag kimutathatóan növeli a kockázati prémiumot.")
            
    except Exception as e:
        st.error(f"Hiba történt a predikció során: {e}")
        st.info("Ellenőrizd, hogy a Colab-ban mentett pkl verziója megegyezik-e a Streamlit környezettel.")