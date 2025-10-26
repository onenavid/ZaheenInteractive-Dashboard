import streamlit as st
import plotly.express as px


def viz_builder(df):
    cols = df.columns.tolist()
    numeric = df.select_dtypes(include=['number']).columns.tolist()

    chart = st.selectbox('Chart', ['Scatter', 'Line', 'Bar', 'Histogram', 'Boxplot', 'Heatmap'])

    x = st.selectbox('X', cols, index=0)
    y = None
    if chart not in ['Histogram', 'Heatmap']:
        y = st.selectbox('Y', cols, index=1 if len(cols) > 1 else 0)

    color = st.selectbox('Color (optional)', [None] + cols)

    if st.button('Generate'):
        fig = None
        if chart == 'Scatter':
            fig = px.scatter(df, x=x, y=y, color=color, title=f'{y} vs {x}')
        elif chart == 'Line':
            fig = px.line(df, x=x, y=y, color=color, title=f'{y} vs {x}')
        elif chart == 'Bar':
            fig = px.bar(df, x=x, y=y, color=color, title=f'{y} by {x}')
        elif chart == 'Histogram':
            fig = px.histogram(df, x=x, color=color, title=f'Distribution of {x}')
        elif chart == 'Boxplot':
            fig = px.box(df, x=x, y=y, color=color, title=f'{y} by {x}')
        elif chart == 'Heatmap':
            if len(numeric) > 1:
                corr = df[numeric].corr()
                fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
            else:
                st.warning('Not enough numeric columns for heatmap.')
                return

        if fig:
            st.plotly_chart(fig, use_container_width=True)
