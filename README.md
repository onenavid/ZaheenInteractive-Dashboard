# Zaheen (ذہین) — Urdu for intelligent or wise.
The app embodies clarity, simplicity, and insight — data exploration without friction.
Zaheen is a comprehensive self-service Business Intelligence and Machine Learning application built with Streamlit. Upload your datasets, filter and preprocess data, visualize patterns, run statistical analyses, build ML models, and export results - all through an intuitive interface.

## Tech Stack

- **Python**
- **Streamlit** for the web interface  
- **Pandas** for data handling  
- **Plotly** for interactive charts  
- **Scikit Learn** for model training

## Features

### Data Upload
- Support for CSV and Excel files (.csv, .xlsx, .xls)
- Automatic data caching for faster interactions
- Intelligent error handling with user-friendly messages
- Instant data preview and shape information

### Data Filtering
- Interactive filtering with categorical pickers
- Numeric range sliders for continuous variables
- Real-time filter application across all tabs
- Filter preview and reset functionality
- Support for up to 50 unique categorical values per column

### Data Preprocessing
- **Missing Value Imputation**: Mean, median, most frequent, or constant strategies
- **Categorical Encoding**: One-hot encoding or label encoding
- **Feature Scaling**: StandardScaler (z-score) or MinMaxScaler (0-1 range)
- Sequential preprocessing pipeline
- Preview preprocessed data before modeling

### Visualization
- Interactive chart builder with Plotly
- Multiple chart types:
  - Scatter plots
  - Line charts
  - Bar charts
  - Histograms
  - Box plots
  - Correlation heatmaps
- Customizable axes and color grouping
- Works with filtered and preprocessed data

### Statistical Analysis
- Descriptive statistics summary
- Missing value detection
- Correlation matrices for numeric columns
- Hypothesis testing (t-tests)
- Automatic updates with filtered/preprocessed data

### Machine Learning
- **Regression**: Random Forest Regressor with R² and RMSE metrics
- **Classification**: Random Forest Classifier with accuracy, precision, recall, and F1 scores
- **Clustering**: K-Means clustering with configurable cluster counts
- Train/test split customization
- Model persistence (save and load trained models)
- Cross-validation support

### Model Explainability
- Built-in feature importance visualization
- Permutation importance with confidence intervals
- Interactive Plotly charts
- Top 10 feature rankings
- Expandable full feature importance tables

### Model Comparison
- Side-by-side comparison of multiple algorithms
- **Regression Models**: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, Decision Tree, KNN, SVM
- **Classification Models**: Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, KNN, SVM
- Cross-validation option (5-fold)
- Automated best model selection
- Comparative visualizations for all metrics
- Progress tracking during training

### Data Export
- Download original, filtered, or preprocessed datasets as CSV
- Export model predictions with actual vs predicted values
- Save model comparison results
- One-click download buttons
- Flexible export options

## Installation

All required dependencies are pre-installed:
- streamlit==1.39.0
- pandas==2.2.3
- plotly==5.24.1
- numpy==1.26.4
- scikit-learn==1.5.2
- scipy==1.14.1
- seaborn==0.13.2
- joblib==1.3.2
- openpyxl

## Usage

The application is configured to run automatically. Simply upload a dataset to begin exploring your data.

### Workflow
1. **Upload**: Click the file uploader and select a CSV or Excel file
2. **Overview**: View dataset shape and preview the first few rows
3. **Filters**: Apply categorical and numeric filters to focus on relevant data
4. **Preprocess**: Handle missing values, encode categories, and scale features
5. **Visualize**: Create interactive charts to explore relationships
6. **Analyze**: Review statistical summaries and correlations
7. **Model**: Build and train individual machine learning models with explainability
8. **Compare Models**: Evaluate multiple algorithms side-by-side
9. **Export**: Download processed data, predictions, and comparison results

## Key Capabilities

- **End-to-End ML Pipeline**: From raw data upload to model deployment
- **No-Code Interface**: Accessible to business users and data scientists alike
- **Session-Based**: No database required - work with your data privately
- **Export Everything**: Download all intermediate and final results
- **Production-Ready**: Built with enterprise-grade libraries (scikit-learn, pandas, plotly)
