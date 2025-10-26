import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go


def compare_models(df):
    """Compare multiple ML models side-by-side."""
    st.write('### Model Comparison Tool')
    st.write('Train and compare multiple algorithms to find the best performer')
    
    cols = df.columns.tolist()
    task = st.selectbox('Task Type', ['Regression', 'Classification'], key='compare_task')
    
    target = st.selectbox('Target Variable', cols, key='compare_target')
    features = st.multiselect('Features (leave empty = use all numeric)', cols, default=None, key='compare_features')
    
    if not features:
        X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
    else:
        X = df[features]
    y = df[target]
    
    X_numeric = X.select_dtypes(include=['number'])
    
    if X_numeric.shape[1] == 0:
        st.error('No numeric features available for modeling')
        return
    
    test_size = st.slider('Test size', 0.1, 0.5, 0.2, key='compare_test_size')
    use_cv = st.checkbox('Use Cross-Validation (5-fold)', value=False)
    
    if task == 'Regression':
        available_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Support Vector Machine': SVR()
        }
        metric_name = 'R²'
    else:
        available_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Support Vector Machine': SVC(random_state=42)
        }
        metric_name = 'Accuracy'
    
    selected_models = st.multiselect(
        'Select models to compare',
        list(available_models.keys()),
        default=list(available_models.keys())[:3],
        key='selected_models'
    )
    
    if st.button('Compare Models', key='compare_button'):
        if not selected_models:
            st.warning('Please select at least one model')
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42
        )
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, model_name in enumerate(selected_models):
            status_text.text(f'Training {model_name}...')
            model = available_models[model_name]
            
            try:
                if use_cv:
                    if task == 'Regression':
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        train_score = cv_scores.mean()
                        train_std = cv_scores.std()
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                        train_score = cv_scores.mean()
                        train_std = cv_scores.std()
                else:
                    model.fit(X_train, y_train)
                    if task == 'Regression':
                        train_score = r2_score(y_train, model.predict(X_train))
                    else:
                        train_score = accuracy_score(y_train, model.predict(X_train))
                    train_std = 0
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if task == 'Regression':
                    test_score = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    results.append({
                        'Model': model_name,
                        'Train R²': train_score,
                        'Test R²': test_score,
                        'RMSE': rmse,
                        'CV Std': train_std if use_cv else None
                    })
                else:
                    test_score = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0.0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0.0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0.0)
                    
                    results.append({
                        'Model': model_name,
                        'Train Accuracy': train_score,
                        'Test Accuracy': test_score,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'CV Std': train_std if use_cv else None
                    })
                
            except Exception as e:
                st.error(f'Error training {model_name}: {str(e)}')
            
            progress_bar.progress((idx + 1) / len(selected_models))
        
        status_text.text('Training complete!')
        progress_bar.empty()
        
        results_df = pd.DataFrame(results)
        
        st.write('### Comparison Results')
        st.dataframe(results_df.style.highlight_max(axis=0, subset=[col for col in results_df.columns if col not in ['Model', 'CV Std', 'RMSE']]))
        
        if task == 'Regression':
            best_model = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
            st.success(f'🏆 Best performing model: **{best_model}** (Test R² = {results_df["Test R²"].max():.4f})')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Train R²'],
                name='Train R²',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Test R²'],
                name='Test R²',
                marker_color='darkblue'
            ))
            fig.update_layout(
                title='Model Comparison: R² Scores',
                xaxis_title='Model',
                yaxis_title='R² Score',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.bar(
                results_df,
                x='Model',
                y='RMSE',
                title='Model Comparison: RMSE (Lower is Better)',
                color='RMSE',
                color_continuous_scale='Reds_r'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            best_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
            st.success(f'🏆 Best performing model: **{best_model}** (Test Accuracy = {results_df["Test Accuracy"].max():.4f})')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Train Accuracy'],
                name='Train Accuracy',
                marker_color='lightgreen'
            ))
            fig.add_trace(go.Bar(
                x=results_df['Model'],
                y=results_df['Test Accuracy'],
                name='Test Accuracy',
                marker_color='darkgreen'
            ))
            fig.update_layout(
                title='Model Comparison: Accuracy Scores',
                xaxis_title='Model',
                yaxis_title='Accuracy',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            metrics_df = results_df[['Model', 'Precision', 'Recall', 'F1 Score']].melt(
                id_vars='Model', var_name='Metric', value_name='Score'
            )
            fig2 = px.bar(
                metrics_df,
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title='Classification Metrics Comparison'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.session_state.comparison_results = results_df
