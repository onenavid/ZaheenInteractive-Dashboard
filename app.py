import streamlit as st
from modules.data_upload import load_data
from modules.data_viz import viz_builder
from modules.statistics import show_statistics
from modules.ml_models import ml_builder
from modules.filters import apply_filters
from modules.preprocessing import preprocess_data
from modules.model_comparison import compare_models
import pandas as pd


st.set_page_config(page_title="Zaheen — BI + ML", layout="wide")

st.title("Zaheen Beta")

st.markdown("""
Zaheen: upload data, visualize, run stats, and build simple ML models.
""")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        if 'use_filtered' not in st.session_state:
            st.session_state.use_filtered = False
        if 'use_preprocessed' not in st.session_state:
            st.session_state.use_preprocessed = False
        
        tabs = st.tabs(["Overview", "Filters", "Preprocess", "Visualize", "Analyze", "Model", "Compare Models", "Export"])

        with tabs[0]:
            st.subheader("Overview")
            st.write(f"Shape: {df.shape}")
            st.dataframe(df.head())

        with tabs[1]:
            st.subheader("Filters")
            filtered_df = apply_filters(df)
            st.session_state.filtered_df = filtered_df
            st.session_state.use_filtered = True
            
            if filtered_df.shape[0] > 0:
                st.write("### Filtered Data Preview")
                st.dataframe(filtered_df.head(10))
            else:
                st.warning("No rows match the current filters. Please adjust your selections.")

        with tabs[2]:
            st.subheader("Preprocess")
            base_df = st.session_state.get('filtered_df', df) if st.session_state.use_filtered else df
            preprocessed_df = preprocess_data(base_df)
            st.session_state.preprocessed_df = preprocessed_df
            st.session_state.use_preprocessed = True

        if st.session_state.use_preprocessed and 'preprocessed_df' in st.session_state:
            working_df = st.session_state.preprocessed_df
        elif st.session_state.use_filtered and 'filtered_df' in st.session_state:
            working_df = st.session_state.filtered_df
        else:
            working_df = df
        
        with tabs[3]:
            st.subheader("Visualize")
            if st.session_state.use_preprocessed and 'preprocessed_df' in st.session_state:
                st.info(f"Using preprocessed data: {st.session_state.preprocessed_df.shape[0]} rows, {st.session_state.preprocessed_df.shape[1]} columns")
            elif st.session_state.use_filtered and 'filtered_df' in st.session_state:
                st.info(f"Using filtered data: {st.session_state.filtered_df.shape[0]} rows")
            viz_builder(working_df)

        with tabs[4]:
            st.subheader("Analyze")
            if st.session_state.use_preprocessed and 'preprocessed_df' in st.session_state:
                st.info(f"Using preprocessed data: {st.session_state.preprocessed_df.shape[0]} rows, {st.session_state.preprocessed_df.shape[1]} columns")
            elif st.session_state.use_filtered and 'filtered_df' in st.session_state:
                st.info(f"Using filtered data: {st.session_state.filtered_df.shape[0]} rows")
            show_statistics(working_df)

        with tabs[5]:
            st.subheader("Model")
            if st.session_state.use_preprocessed and 'preprocessed_df' in st.session_state:
                st.info(f"Using preprocessed data: {st.session_state.preprocessed_df.shape[0]} rows, {st.session_state.preprocessed_df.shape[1]} columns")
            elif st.session_state.use_filtered and 'filtered_df' in st.session_state:
                st.info(f"Using filtered data: {st.session_state.filtered_df.shape[0]} rows")
            ml_builder(working_df)

        with tabs[6]:
            st.subheader("Compare Models")
            if st.session_state.use_preprocessed and 'preprocessed_df' in st.session_state:
                st.info(f"Using preprocessed data: {st.session_state.preprocessed_df.shape[0]} rows, {st.session_state.preprocessed_df.shape[1]} columns")
            elif st.session_state.use_filtered and 'filtered_df' in st.session_state:
                st.info(f"Using filtered data: {st.session_state.filtered_df.shape[0]} rows")
            compare_models(working_df)

        with tabs[7]:
            st.subheader("Export Data")
            
            st.write("### Download Processed Data")
            
            export_options = st.multiselect(
                "Select data to export:",
                ["Original Data", "Filtered Data", "Preprocessed Data"],
                default=["Original Data"]
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "Original Data" in export_options:
                    csv_original = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Original Data",
                        data=csv_original,
                        file_name="zaheen_original_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if "Filtered Data" in export_options and 'filtered_df' in st.session_state:
                    csv_filtered = st.session_state.filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Filtered Data",
                        data=csv_filtered,
                        file_name="zaheen_filtered_data.csv",
                        mime="text/csv"
                    )
                elif "Filtered Data" in export_options:
                    st.info("No filtered data available. Visit the Filters tab first.")
            
            with col3:
                if "Preprocessed Data" in export_options and 'preprocessed_df' in st.session_state:
                    csv_preprocessed = st.session_state.preprocessed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Preprocessed Data",
                        data=csv_preprocessed,
                        file_name="zaheen_preprocessed_data.csv",
                        mime="text/csv"
                    )
                elif "Preprocessed Data" in export_options:
                    st.info("No preprocessed data available. Visit the Preprocess tab first.")
            
            st.write("### Download Model Results")
            
            if 'model_predictions' in st.session_state and 'X_test' in st.session_state:
                results_df = st.session_state.X_test.copy()
                results_df['Predicted'] = st.session_state.model_predictions
                if 'y_test' in st.session_state:
                    results_df['Actual'] = st.session_state.y_test.values
                
                csv_predictions = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Model Predictions",
                    data=csv_predictions,
                    file_name="zaheen_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.info("No model predictions available. Train a model in the Model tab first.")
            
            if 'comparison_results' in st.session_state:
                csv_comparison = st.session_state.comparison_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Model Comparison Results",
                    data=csv_comparison,
                    file_name="zaheen_model_comparison.csv",
                    mime="text/csv"
                )
            else:
                st.info("No comparison results available. Compare models in the Compare Models tab first.")

else:
    st.info("Upload a dataset to begin (CSV or Excel).")
