import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import app_utils
import numpy as np

st.set_page_config(page_title="Feature Importance", page_icon="📈")

st.title("📈 Feature Importance Analysis")
st.markdown("Understand which features drive computer prices the most.")

# Load Data & Model
try:
    df = app_utils.load_data()
    model, feature_info, input_options = app_utils.load_model_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# 1. Global Feature Importance (from Model)
st.subheader("Global Feature Importance")
st.markdown("This chart shows the overall importance of each feature in the Gradient Boosting model.")

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    features = feature_info['selected_features']
    
    # Create DataFrame
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=True)
    
    fig_bar = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.warning("Model does not expose feature importances.")

# 2. SHAP Summary Plot
st.subheader("SHAP Summary Plot")
st.markdown("This plot shows the impact of each feature on the model output. Points are colored by feature value (Red = High, Blue = Low).")

if st.button("Generate SHAP Summary"):
    with st.spinner("Calculating SHAP values (this may take a moment)..."):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Starting SHAP summary generation...")
        
        # Use a subset of data for speed
        subset_df = df.sample(min(200, len(df)), random_state=42)
        
        # Preprocess subset
        X_batch = app_utils.preprocess_batch(subset_df, feature_info)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_batch)
        
        logger.info("SHAP values calculated successfully. Generating plot...")
        
        # Display Summary Plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_batch, show=False)
        st.pyplot(fig)
        
        logger.info("SHAP summary plot generated and displayed.")
