# Insurance Cost Prediction

## Overview
This project analyzes an insurance dataset to predict insurance charges based on various factors such as age, BMI, smoking status, and region. It uses exploratory data analysis (EDA) techniques and implements a Random Forest Regressor model to make predictions.

## Dataset
The dataset used is 'insurance.csv'(https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset), which contains the following features:
- age
- sex
- bmi
- children
- smoker
- region
- charges (target variable)

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Data Preprocessing
1. Loaded the dataset using pandas
2. Encoded categorical variables:
   - 'sex' (male: 1, female: 0)
   - 'smoker' (yes: 1, no: 0)
   - 'region' (one-hot encoded)

## Exploratory Data Analysis (EDA)
- Visualized the distribution of insurance charges
- Created scatter plots to show relationships between age, BMI, and charges
- Generated histograms for all features
- Produced a correlation heatmap

## Machine Learning Model
- Used Random Forest Regressor
- Split the data into training (80%) and testing (20%) sets
- Performed initial model training and evaluation
- Conducted hyperparameter tuning using GridSearchCV

## Model Evaluation
- Calculated R-squared score
- Computed Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
- Visualized predicted vs actual charges

## Feature Importance
- Analyzed and visualized the importance of each feature in predicting insurance charges

## Viewing the Notebook  
To view the Jupyter Notebook online without downloading it, use the following nbviewer link:  
[View the Notebook](https://nbviewer.org/github/Rafal852/Insurance-Premium-Prediction/blob/main/Insurance_prediciton.ipynb)


## Installation and Usage  

### Install Dependencies  
Ensure you have Python installed and run the following command:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn


