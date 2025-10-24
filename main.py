import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Simple Data Dashboard", layout="wide")

st.title("Zaheen - Simple Interactive Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Variable selection
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Scatter", "Line", "Bar", "Histogram", "Boxplot", "Heatmap"]
    )

    x = st.selectbox("X-axis", all_cols)
    y = None
    if chart_type not in ["Histogram", "Heatmap"]:
        y = st.selectbox("Y-axis", all_cols)

    color = st.selectbox("Color (optional)", [None] + all_cols)

    # Plot
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=f"{y} vs {x}")
    elif chart_type == "Line":
        fig = px.line(df, x=x, y=y, color=color, title=f"{y} vs {x}")
    elif chart_type == "Bar":
        fig = px.bar(df, x=x, y=y, color=color, title=f"{y} by {x}")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x, color=color, title=f"Distribution of {x}")
    elif chart_type == "Boxplot":
        fig = px.box(df, x=x, y=y, color=color, title=f"{y} by {x}")
    elif chart_type == "Heatmap":
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        else:
            st.warning("Not enough numeric columns for a heatmap.")
            st.stop()

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a CSV or Excel file to begin.")
