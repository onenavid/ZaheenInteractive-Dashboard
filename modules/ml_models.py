import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
import joblib


def ml_builder(df):
    st.write('ML builder — pick task type and target')
    task = st.selectbox('Task', ['Regression', 'Classification', 'Clustering'])
    cols = df.columns.tolist()

    if task in ['Regression', 'Classification']:
        target = st.selectbox('Target', cols)
        features = st.multiselect('Features (leave empty = use all numeric)', cols, default=None)

        if not features:
            X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
        else:
            X = df[features]
        y = df[target]

        test_size = st.slider('Test size', 0.1, 0.5, 0.2)
        
        show_explainability = st.checkbox('Show Model Explainability', value=True)
        
        if st.button('Train'):
            X_numeric = X.select_dtypes(include=['number'])
            X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=test_size, random_state=42)

            if task == 'Regression':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.write('### Model Performance')
                st.write('R2:', r2_score(y_test, preds))
                st.write('RMSE:', np.sqrt(mean_squared_error(y_test, preds)))

            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.write('### Model Performance')
                st.write('Accuracy:', accuracy_score(y_test, preds))
                st.text(classification_report(y_test, preds))

            if show_explainability:
                st.write('### Model Explainability')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('#### Feature Importance (Built-in)')
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': X_numeric.columns,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Features by Importance'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View all feature importances"):
                            st.dataframe(importance_df)
                    else:
                        st.info('Feature importance not available for this model type')
                
                with col2:
                    st.write('#### Permutation Importance')
                    try:
                        perm_importance = permutation_importance(
                            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
                        )
                        
                        perm_df = pd.DataFrame({
                            'Feature': X_numeric.columns,
                            'Importance': perm_importance.importances_mean,
                            'Std': perm_importance.importances_std
                        }).sort_values('Importance', ascending=False)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=perm_df.head(10)['Feature'],
                            x=perm_df.head(10)['Importance'],
                            error_x=dict(type='data', array=perm_df.head(10)['Std']),
                            orientation='h',
                            name='Permutation Importance'
                        ))
                        fig.update_layout(
                            title='Top 10 Features by Permutation Importance',
                            xaxis_title='Importance',
                            yaxis_title='Feature'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("View all permutation importances"):
                            st.dataframe(perm_df)
                    except Exception as e:
                        st.error(f'Permutation importance calculation failed: {e}')

            st.session_state.trained_model = model
            st.session_state.model_predictions = preds
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            if st.checkbox('Save model'):
                fname = st.text_input('Model filename', value='zaheen_model.joblib')
                joblib.dump(model, fname)
                st.success(f'Model saved to {fname}')

    else:
        numeric = df.select_dtypes(include=['number'])
        k = st.slider('n_clusters', 2, 10, 3)
        if st.button('Run KMeans'):
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(numeric.fillna(0))
            st.write('Cluster counts')
            st.write(pd.Series(labels).value_counts())
            st.write('Centroids')
            st.write(km.cluster_centers_)
