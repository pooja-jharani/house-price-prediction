"""Interactive demo: ``streamlit run app/streamlit_app.py`` (requires ``pip install -e .[app]``)."""

import json

import streamlit as st

from house_price import config
from house_price.predict import HouseFeatures, predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Predictor")
st.caption("Trained on a synthetic dataset. For demonstration, not real valuation.")

if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `house-price train` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    area = st.slider("Area (sq ft)", 400, 4500, 1500, step=50)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
with col2:
    age = st.slider("Age (years)", 0, 80, 10)
    distance = st.slider("Distance to city centre (km)", 0.2, 50.0, 8.0, step=0.2)
    location = st.selectbox("Location", config.LOCATIONS, index=1)

if st.button("Predict price", type="primary"):
    try:
        features = HouseFeatures(area, bedrooms, bathrooms, age, distance, location)
        st.metric("Estimated price", f"{config.CURRENCY} {predict_price(features):,.0f}")
    except ValueError as exc:
        st.error(str(exc))

if config.METADATA_PATH.exists():
    meta = json.loads(config.METADATA_PATH.read_text())
    with st.expander("About the model"):
        m = meta["test_metrics"]
        st.write(f"**Model:** {meta['model_name']}  |  **Trained:** {meta['trained_at']}")
        st.write(f"**Test R²:** {m['r2']:.3f}  |  **MAE:** {config.CURRENCY} {m['mae']:,.0f}"
                 f"  |  **MAPE:** {m['mape']:.1%}")
