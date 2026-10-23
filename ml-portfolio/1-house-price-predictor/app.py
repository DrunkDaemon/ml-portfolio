import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor")
st.title("California House Price Predictor")
bundle = joblib.load("model.joblib")
model, stats, metrics = bundle["model"], bundle["stats"], bundle["metrics"]

labels = {
    "MedInc": "Median income (x $10k)", "HouseAge": "House age (years)",
    "AveRooms": "Avg rooms", "AveBedrms": "Avg bedrooms", "Population": "Block population",
    "AveOccup": "Avg occupants per house", "Latitude": "Latitude", "Longitude": "Longitude",
}
values = {}
for col, (lo, med, hi) in stats.items():
    values[col] = st.sidebar.slider(labels[col], float(lo), float(hi), float(med))

price = model.predict(pd.DataFrame([values]))[0] * 100_000
st.metric("Estimated median house value", f"${price:,.0f}")
st.caption(f"Model test error: about ${metrics['MAE_usd']:,.0f} MAE, R² = {metrics['R2']:.2f}")
st.map(pd.DataFrame({"lat": [values["Latitude"]], "lon": [values["Longitude"]]}))
