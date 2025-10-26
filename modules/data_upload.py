import pandas as pd
import streamlit as st


@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel into a DataFrame. Returns pd.DataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
        else:
            st.error(f"Unsupported file format: {name}. Please upload CSV, XLSX, or XLS files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None
