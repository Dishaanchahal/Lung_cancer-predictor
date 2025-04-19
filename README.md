# Lung_cancer-predictor
This is a very simple project in which I have taken Lung cancer data from Kaggle and used it to train a Naive Baye's classifier which will ask user input and then predict whether the user has a high or low risk of Lung cancer.

## Overview
This program uses a machine learning model trained on lung cancer data to provide a personalized risk assessment. It evaluates various factors known to be associated with lung cancer and assigns weights to each factor based on their predictive importance.

## Disclaimer
**This tool is for educational purposes only and is NOT a medical diagnostic tool.** The results should not be used to diagnose or treat any medical condition. Always consult with a healthcare professional for proper medical evaluation and advice.

## Features
- Predicts lung cancer risk based on multiple health factors
- Assigns weights to each factor based on its predictive importance
- Provides a weighted risk score
- Identifies top contributing factors to your risk assessment

## Requirements
- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - scikit-learn

## Installation
1. Ensure you have Python installed on your system
2. Install required packages:
   ```
   pip install pandas numpy scikit-learn
   ```
3. Place your data file (paste.txt) in the same directory as the program

## Usage
1. Run the program:
   ```
   python lung_cancer_risk_assessment.py
   ```
2. Answer the questions about your health factors
3. Review your risk assessment results

## Input Factors
The program will ask about these health factors:
- Gender
- Age
- Smoking status
- Yellow fingers
- Anxiety
- Peer pressure
- Chronic disease
- Fatigue
- Allergies
- Wheezing
- Alcohol consumption
- Coughing
- Shortness of breath
- Swallowing difficulty
- Chest pain

## Understanding Results
The program provides:
- A binary prediction (risk detected or low risk)
- A probability percentage based on the model
- A weighted risk score (0-100)
- The top factors contributing to your risk assessment

## Data Source
The model is trained on a dataset of patient records with confirmed lung cancer diagnoses from kaggle. The data includes demographic information and various symptoms associated with lung cancer.

## Limitations
- The assessment is based on statistical patterns in the training data
- Individual medical conditions can vary significantly
- Many other factors not included in this assessment can influence cancer risk
- The model cannot distinguish between correlation and causation

## Author
Dishaan Chahal
