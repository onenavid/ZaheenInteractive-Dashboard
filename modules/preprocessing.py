import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    """Provide preprocessing options for the dataset."""
    st.write("### Data Preprocessing")
    
    preprocessed_df = df.copy()
    
    st.write(f"Original shape: {df.shape}")
    
    preprocessing_options = st.multiselect(
        "Select preprocessing steps (applied in order):",
        ["Handle Missing Values", "Encode Categorical Variables", "Scale Numeric Features"]
    )
    
    if "Handle Missing Values" in preprocessing_options:
        st.write("#### Missing Value Imputation")
        
        numeric_cols = preprocessed_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = preprocessed_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            numeric_strategy = st.selectbox(
                "Numeric imputation strategy",
                ["mean", "median", "most_frequent", "constant"],
                key="numeric_impute"
            )
            
            if numeric_strategy == "constant":
                fill_value = st.number_input("Fill value for numeric", value=0.0)
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            else:
                imputer = SimpleImputer(strategy=numeric_strategy)
            
            preprocessed_df[numeric_cols] = imputer.fit_transform(preprocessed_df[numeric_cols])
            st.success(f"Imputed {len(numeric_cols)} numeric columns using '{numeric_strategy}' strategy")
        
        if categorical_cols:
            categorical_strategy = st.selectbox(
                "Categorical imputation strategy",
                ["most_frequent", "constant"],
                key="categorical_impute"
            )
            
            if categorical_strategy == "constant":
                fill_value = st.text_input("Fill value for categorical", value="missing")
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            else:
                imputer = SimpleImputer(strategy=categorical_strategy)
            
            preprocessed_df[categorical_cols] = imputer.fit_transform(preprocessed_df[categorical_cols].astype(str))
            st.success(f"Imputed {len(categorical_cols)} categorical columns using '{categorical_strategy}' strategy")
    
    if "Encode Categorical Variables" in preprocessing_options:
        st.write("#### Categorical Encoding")
        
        categorical_cols = preprocessed_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            encoding_method = st.radio(
                "Encoding method",
                ["One-Hot Encoding", "Label Encoding"],
                key="encoding_method"
            )
            
            cols_to_encode = st.multiselect(
                "Select columns to encode",
                categorical_cols,
                default=categorical_cols[:min(5, len(categorical_cols))],
                key="cols_to_encode"
            )
            
            if cols_to_encode:
                if encoding_method == "One-Hot Encoding":
                    preprocessed_df = pd.get_dummies(
                        preprocessed_df,
                        columns=cols_to_encode,
                        drop_first=True,
                        prefix=cols_to_encode
                    )
                    st.success(f"One-hot encoded {len(cols_to_encode)} columns")
                else:
                    for col in cols_to_encode:
                        le = LabelEncoder()
                        preprocessed_df[col] = le.fit_transform(preprocessed_df[col].astype(str))
                    st.success(f"Label encoded {len(cols_to_encode)} columns")
        else:
            st.info("No categorical columns to encode")
    
    if "Scale Numeric Features" in preprocessing_options:
        st.write("#### Feature Scaling")
        
        numeric_cols = preprocessed_df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            scaling_method = st.radio(
                "Scaling method",
                ["StandardScaler (mean=0, std=1)", "MinMaxScaler (range 0-1)"],
                key="scaling_method"
            )
            
            cols_to_scale = st.multiselect(
                "Select columns to scale",
                numeric_cols,
                default=numeric_cols,
                key="cols_to_scale"
            )
            
            if cols_to_scale:
                if "StandardScaler" in scaling_method:
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                preprocessed_df[cols_to_scale] = scaler.fit_transform(preprocessed_df[cols_to_scale])
                st.success(f"Scaled {len(cols_to_scale)} columns using {scaling_method.split()[0]}")
        else:
            st.info("No numeric columns to scale")
    
    st.write(f"**Preprocessed shape: {preprocessed_df.shape}**")
    st.write(f"**Missing values remaining: {preprocessed_df.isna().sum().sum()}**")
    
    if st.checkbox("Show preprocessed data preview"):
        st.dataframe(preprocessed_df.head(10))
    
    return preprocessed_df
