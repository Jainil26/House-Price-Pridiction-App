import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Train and cache model so it only runs once
@st.cache_resource
def load_model():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

model, scaler = load_model()

st.title("House Price Predictor")
st.write("Adjust the values and click Predict.")

med_inc    = st.slider("Median Income (x$10k)", 0.5, 15.0, 3.0)
house_age  = st.slider("House Age (years)",      1.0, 52.0, 20.0)
ave_rooms  = st.slider("Avg Rooms",              1.0, 10.0, 5.0)
ave_bedrms = st.slider("Avg Bedrooms",           1.0, 5.0,  1.0)
population = st.slider("Population",             3.0, 3500.0, 1000.0)
ave_occup  = st.slider("Avg Occupants",          1.0, 6.0,  3.0)
latitude   = st.slider("Latitude",               32.0, 42.0, 35.0)
longitude  = st.slider("Longitude",              -125.0, -114.0, -119.0)

if st.button("Predict Price"):
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
    population, ave_occup, latitude, longitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    st.success(f"Estimated Price: ${prediction * 100_000:,.0f}")