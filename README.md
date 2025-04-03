# House Price Prediction Project - India

## Goal
This project demonstrates a comprehensive end-to-end machine learning pipeline for predicting house prices in India. It covers the entire lifecycle of a machine learning project, from data collection to production deployment, serving as both an educational resource and a practical implementation.

## Project Structure

### Notebooks
The project follows a sequential approach through several Jupyter notebooks:

- `data_collection.ipynb`:
  - Data wrangling and preparation
  - Adding city names from filenames to each dataframe
  - Joining all dataframes into a unified dataset
  - Train/test splitting
  - Saving processed datasets

- `descriptive_analysis.ipynb`:
  - Exploratory data analysis
  - Statistical summaries
  - Data visualization
  - Pattern identification

- `base_model.ipynb`:
  - Step-by-step development of a heuristic baseline model
  - Price prediction based on location and price per square foot
  - Benchmark performance establishment

- `preprocessing_data.ipynb`:
  - Missing value imputation
  - Low and high cardinality categorical feature transformation
  - Numerical feature normalization and standardization
  - Cross-feature creation

- `feature_selection.ipynb`:
  - Feature importance analysis
  - Dimensionality reduction
  - Optimal feature subset selection

- `model_comparison_01.ipynb`: 
  - Train a Random Forest model
  - Evaluate model performance

- `model_comparison_02.ipynb`:
  - Train a HistGradientBoostingRegressor model
  - Evaluate model performance

### Refactored Notebooks
Each of the main notebooks has a corresponding `_refactored.ipynb` version. These refactored notebooks perform the same tasks as their non-refactored counterparts but utilize the encapsulated code from the `models` and `utils` folders. This approach demonstrates how to structure code for better reusability and maintainability:

- `data_collection_refactored.ipynb`
- `base_model_refactored.ipynb`
- `preprocessed_refactored.ipynb`: 
  - Combines preprocessing, feature selection, and model training
  - Utilizes classes and functions from `utils` and `models` for streamlined operations
  - Demonstrates modular code application for efficient data processing and model development

These refactored notebooks highlight the benefits of using modular code, such as cleaner notebook code, easier maintenance, and improved testing capabilities.

### Modular Code Structure
The core functionality has been refactored into organized Python modules:

#### `/models`
Contains the implementation of model classes:
- `base_model.py`: Implementation of the benchmark model using location and price/sqft rules
- [Other model implementations]

#### `/utils`
Contains utility functions and helper classes used across the project:
- Data preprocessing utilities
- Feature engineering functions
- Model evaluation metrics
- [Other utility modules]

#### `/api`
- Flask API implementation for model serving
- Streamlit dashboard for interactive predictions

## Code Organization

### Notebook vs Module Versions
This project maintains two complementary implementations:

1. **Jupyter Notebooks**: 
   - Interactive exploration and visualization
   - Step-by-step documentation of the ML pipeline
   - Educational resource for understanding each phase
   - Experimental analysis and results visualization

2. **Refactored Modules**: 
   - Production-ready code organization
   - Encapsulated functionality in classes and functions
   - Improved maintainability and testing capability
   - Used in deployment and production scenarios

### Project Pipeline Phases
1. Data Collection and Wrangling
2. Exploratory Data Analysis
3. Benchmark Model Development (Rule-based)
4. Data Preprocessing and Transformation
5. Feature Selection and Engineering
6. Model Selection and Training
7. Hyperparameter Optimization
8. Model Evaluation and Uncertainty Analysis
9. Model Explainability
10. API Development and Deployment
11. Production Monitoring and Drift Detection

## Getting Started
[Installation instructions]

## Dependencies
- Python 3.8+
- Scikit-learn
- Pandas
- NumPy
- Flask
- Streamlit
[Other dependencies]

## Usage
### Local Development
[Instructions for running notebooks and local development]

### API Deployment
[Instructions for deploying the Flask API]

### Dashboard
[Instructions for running the Streamlit dashboard]

## Production Monitoring
[Description of monitoring setup and drift detection]