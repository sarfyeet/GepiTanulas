import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥", layout="centered")

st.title("🏥 Egészségbiztosítási Díjkalkulátor")
st.write("Ez az alkalmazás egy Scikit-Learn Pipeline és Random Forest Regressor segítségével, valós adatok alapján becsüli meg az éves biztosítási díjat.")

# Modell betanítása helyben a szerveren (Cache-elve, hogy csak egyszer fusson le)
@st.cache_resource
def train_model_live():
    csv_path = "insurance.csv"
    if not os.path.exists(csv_path):
        st.error(f"Hiba: A '{csv_path}' fájl nem található a GitHub repódban! Kérlek, töltsd fel az app.py mellé.")
        return None
    
    # Adatok beolvasása és tisztítása
    df = pd.read_csv(csv_path).dropna()
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    # Csővezeték (Pipeline) felépítése
    numerical_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1))
    ])
    
    # Tanítás helyben
    pipeline.fit(X, y)
    return pipeline

# Modell betöltése/tanítása
pipeline = train_model_live()

if pipeline is not None:
    st.divider()
    st.subheader("Kérlek, add meg az ügyfél adatait:")

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

    if st.button("Biztosítási díj kiszámítása", type="primary"):
        input_data = pd.DataFrame([{
            'age': age, 'sex': sex, 'bmi': bmi,
            'children': children, 'smoker': smoker, 'region': region
        }])
        
        try:
            predicted_charge = pipeline.predict(input_data)[0]
            st.subheader("A modell által becsült éves költség:")
            st.success(f"💰 **${predicted_charge:,.2f}** / év")
            
            if smoker == "yes":
                st.warning("⚠️ **Elemzési megjegyzés:** A modell kiemelten magas díjat számolt fel, mivel a dohányzás a legmeghatározóbb feature a döntési folyamatban.")
            elif bmi >= 30:
                st.info("💡 **Elemzési megjegyzés:** A 30 feletti BMI érték (elhízás kategória) statisztikailag kimutathatóan növeli a kockázati prémiumot.")
                
        except Exception as e:
            st.error(f"Hiba történt a predikció során: {e}")
