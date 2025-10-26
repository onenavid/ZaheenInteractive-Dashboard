import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats


def show_statistics(df):
    st.write('### Summary')
    st.write(df.describe(include='all'))

    st.write('### Missing values')
    st.write(df.isna().sum())

    numeric = df.select_dtypes(include=['number']).columns.tolist()
    if numeric:
        st.write('### Correlations')
        st.write(df[numeric].corr())

    if len(numeric) >= 2:
        st.write('### Example t-test (first two numeric cols)')
        a, b = df[numeric].iloc[:, 0].dropna(), df[numeric].iloc[:, 1].dropna()
        try:
            tstat, pvalue = stats.ttest_ind(a, b, equal_var=False)
            st.write({'t-stat': float(tstat), 'p-value': float(pvalue)})
        except Exception as e:
            st.write('T-test failed:', e)
