import streamlit as st
import pandas as pd


def apply_filters(df):
    """Apply filters to dataframe based on user selections."""
    filtered_df = df.copy()
    
    st.write("### Filter Data")
    st.write(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    handle_missing = st.checkbox("Exclude rows with missing values in filtered columns", value=False, key="handle_missing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if numeric_cols:
            st.write("#### Numeric Filters")
            for col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                if pd.isna(min_val) or pd.isna(max_val):
                    st.info(f"{col} contains only missing values")
                    continue
                
                if min_val != max_val:
                    selected_range = st.slider(
                        f"{col}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key=f"slider_{col}"
                    )
                    
                    if selected_range != (min_val, max_val):
                        if handle_missing:
                            filtered_df = filtered_df[
                                (filtered_df[col] >= selected_range[0]) & 
                                (filtered_df[col] <= selected_range[1])
                            ]
                        else:
                            mask = (
                                (filtered_df[col] >= selected_range[0]) & 
                                (filtered_df[col] <= selected_range[1])
                            ) | filtered_df[col].isna()
                            filtered_df = filtered_df[mask]
    
    with col2:
        if categorical_cols:
            st.write("#### Categorical Filters")
            for col in categorical_cols:
                unique_vals = df[col].dropna().unique().tolist()
                has_missing = df[col].isna().any()
                
                if has_missing:
                    unique_vals_display = unique_vals + ["(Missing)"]
                else:
                    unique_vals_display = unique_vals
                
                if len(unique_vals) > 0 and len(unique_vals) <= 50:
                    default_selection = unique_vals_display if not handle_missing else unique_vals
                    
                    selected_vals = st.multiselect(
                        f"{col}",
                        options=unique_vals_display,
                        default=default_selection,
                        key=f"multi_{col}"
                    )
                    
                    if selected_vals and set(selected_vals) != set(unique_vals_display):
                        include_missing = "(Missing)" in selected_vals
                        actual_vals = [v for v in selected_vals if v != "(Missing)"]
                        
                        if include_missing and actual_vals:
                            mask = filtered_df[col].isin(actual_vals) | filtered_df[col].isna()
                            filtered_df = filtered_df[mask]
                        elif include_missing:
                            filtered_df = filtered_df[filtered_df[col].isna()]
                        else:
                            filtered_df = filtered_df[filtered_df[col].isin(actual_vals)]
                elif len(unique_vals) > 50:
                    st.info(f"{col} has {len(unique_vals)} unique values - too many to display")
    
    st.write(f"**Filtered dataset: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns**")
    
    if st.button("Reset Filters"):
        st.rerun()
    
    return filtered_df
