# Customer Churn Analysis and Prediction

## Project Overview

This project analyses customer churn using a credit card customer dataset and applies statistical analysis and supervised machine learning to identify patterns associated with customer attrition.

The project follows a complete data analytics workflow, from data extraction and cleaning through exploratory data analysis, hypothesis testing, machine learning, model evaluation and business recommendations.

The aim is to demonstrate how customer data can be transformed into useful insights that could support customer retention and decision-making.

## Business Problem

Customer churn can affect customer relationships and business revenue. Identifying patterns associated with customers who leave can help a business understand customer behaviour and develop more targeted retention strategies.

This project investigates customer demographics, account characteristics and transaction behaviour to understand which factors are associated with churn and whether machine learning can be used to predict customer attrition.

## Project Objectives

The project aims to:

* Explore customer demographics, account activity and transaction behaviour.
* Clean and prepare the dataset for analysis and machine learning.
* Investigate patterns and relationships between customer characteristics and churn.
* Use statistical analysis and hypothesis testing to examine differences between existing and attrited customers.
* Develop and compare supervised machine learning models to predict customer churn.
* Evaluate model performance using accuracy, precision, recall, F1-score and ROC-AUC.
* Identify the features that are most important to churn prediction.
* Produce data-driven insights and recommendations that could support customer retention strategies.

## Analytical Questions

The analysis addresses the following questions:

1. What customer characteristics and behaviours are associated with higher levels of churn?
2. Are there noticeable differences in transaction activity between existing and attrited customers?
3. Is there a statistically significant difference in transaction frequency between existing and attrited customers?
4. Which customer, account and transaction features are most important when predicting customer churn?
5. How effectively can machine learning models predict whether a customer is likely to churn?
6. Which of the selected machine learning models provides stronger predictive performance based on the evaluation metrics used?

## Dataset

The project uses the **BankChurners** credit card customer dataset.

The original dataset contains **10,127 customer records and 23 columns**, covering:

* Customer demographics
* Education and marital status
* Income category
* Card category
* Account information
* Customer relationship information
* Transaction activity
* Customer inactivity and contact information
* Credit information
* Customer attrition status

The target variable is `Attrition_Flag`, which identifies customers as either:

* Existing Customer
* Attrited Customer

The dataset contains:

* **8,500 existing customers**
* **1,627 attrited customers**

This means the target classes are imbalanced, with attrited customers representing approximately 16.1% of the dataset.

## ETL and Data Preparation

The project follows an Extract, Transform and Load (ETL) process.

### Extract

The original customer churn dataset was loaded into Python using pandas.

### Transform

The data was inspected for:

* Missing values
* Duplicate records
* Unexpected values
* Numerical ranges
* Categorical values
* Potentially inappropriate features
* Potential data leakage

No duplicate rows were identified.

The `CLIENTNUM` identifier was removed because it is a customer identifier and does not provide meaningful predictive information.

Two Naive Bayes classifier columns included in the original dataset were also removed because they contain model-generated information related to the attrition outcome. Keeping these variables could introduce data leakage and produce misleading model performance.

Categorical values recorded as `Unknown` were retained because they represent information contained in the original dataset.

The target variable was transformed into a binary `Churn` variable:

* `0` = Existing Customer
* `1` = Attrited Customer

The cleaned dataset contains **20 columns** after the identifier and two model-generated columns were removed.

### Load

The transformed dataset was saved as:

`customer_churn_cleaned.csv`

This creates a reusable cleaned dataset for further analysis and modelling.

## Exploratory Data Analysis

Exploratory data analysis was used to investigate customer demographics, account characteristics and transaction behaviour.

The analysis included:

* Attrition rates by gender
* Attrition rates by income category
* Attrition rates by customer inactivity
* Attrition rates by contact frequency
* Attrition rates by transaction count
* Attrition rates by transaction amount
* Attrition rates by card category
* Attrition rates by customer age
* Attrition rates by total relationship count
* Attrition rates by revolving balance
* Attrition rates by utilisation ratio
* Attrition rates by credit limit
* Attrition rates by months on book
* Correlation analysis of numerical variables

The analysis showed that transaction and engagement-related variables provided useful information about differences between existing and attrited customers.

## Statistical Analysis and Hypothesis Testing

A hypothesis test was conducted to examine whether transaction frequency differed between existing and attrited customers.

### Hypotheses

**Null hypothesis (H0):**

There is no statistically significant difference in average transaction count between existing and attrited customers.

**Alternative hypothesis (H1):**

There is a statistically significant difference in average transaction count between existing and attrited customers.

A Welch independent samples t-test was used.

Results:

* Existing customers' mean transaction count: **68.67**
* Attrited customers' mean transaction count: **44.93**
* t-statistic: **54.14**
* p-value: extremely small and displayed as `0.0` by Python

At the 5% significance level, the null hypothesis was rejected.

The result indicates a statistically significant difference in transaction frequency between the two groups. This demonstrates an association between transaction activity and customer attrition, but it does not establish that lower transaction activity causes churn.

## Machine Learning

This project uses supervised machine learning because the target outcome, customer churn, is known for each customer.

Two classification algorithms were selected:

### Logistic Regression

Logistic Regression was used as a baseline model because it is suitable for binary classification and provides interpretable coefficients.

### Random Forest

Random Forest was selected because it can capture non-linear relationships and interactions between features. It also provides feature importance information that can help identify variables contributing to predictions.

### Data Preparation for Machine Learning

The dataset was divided into:

* **80% training data**
* **20% testing data**

Stratified sampling was used to maintain a similar proportion of existing and attrited customers in both datasets.

Categorical variables were transformed using one-hot encoding.

Numerical variables were standardised using `StandardScaler`.

The preprocessing transformer was fitted only on the training data before being applied to the test data to reduce the risk of data leakage.

The Random Forest model used class weighting to help address the imbalance between existing and attrited customers.

## Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Logistic Regression

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 89.98% |
| Precision | 76.75% |
| Recall    | 53.85% |
| F1-score  | 63.29% |
| ROC-AUC   | 91.67% |

### Random Forest

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 95.46% |
| Precision | 85.85% |
| Recall    | 85.85% |
| F1-score  | 85.85% |
| ROC-AUC   | 98.46% |

The Random Forest model produced stronger results across the evaluation metrics on the test dataset.

These results should be interpreted as the performance of the models on this particular train-test split rather than as a guarantee of performance on future customer data.

## Feature Importance

The Random Forest feature importance analysis identified the following features among the most important predictors:

1. `Total_Trans_Amt`
2. `Total_Trans_Ct`
3. `Total_Revolving_Bal`
4. `Total_Ct_Chng_Q4_Q1`
5. `Avg_Utilization_Ratio`
6. `Total_Amt_Chng_Q4_Q1`
7. `Total_Relationship_Count`
8. `Avg_Open_To_Buy`
9. `Months_Inactive_12_mon`
10. `Credit_Limit`

Transaction-related variables were particularly prominent in the model.

Feature importance represents predictive association and does not demonstrate that a feature causes customer churn.

## Key Business Insights

### Transaction Activity

Attrited customers had a substantially lower average transaction count than existing customers. Transaction count was also one of the most important features in the Random Forest model.

This suggests that changes in transaction activity could be useful as an indicator of customer engagement.

### Transaction Behaviour

Total Transaction Amount and Total Transaction Count were the two most important features in the Random Forest model.

This indicates that transaction behaviour provided substantial predictive information in this dataset.

### Card Utilisation

Customers with very low average utilisation showed higher observed attrition rates in the exploratory analysis.

This may indicate reduced engagement, although the analysis does not demonstrate that low utilisation causes churn.

### Inactivity and Contact Patterns

Customer inactivity and contact frequency showed associations with observed attrition.

These variables may therefore be useful as part of a broader customer monitoring process rather than being considered independently.

## Business Recommendations

Based on the analysis, a business could consider:

1. Monitoring sustained reductions in customer transaction activity as an early engagement signal.
2. Combining transaction activity, inactivity, utilisation and other relevant information when identifying customers who may require further review.
3. Using customer segmentation to support more targeted retention initiatives.
4. Using predictive modelling as a decision-support tool rather than allowing the model to automatically determine customer-facing actions.
5. Validating and monitoring the model regularly before considering real-world deployment.

These recommendations are based on observed associations and predictive patterns in the dataset. They should not be interpreted as evidence of causal relationships.

## Limitations

Several limitations should be considered.

### Dataset Imbalance

The dataset contains substantially more existing customers than attrited customers. Although class weighting was used for the Random Forest model, additional approaches such as resampling could be investigated.

### Model Validation

The models were evaluated using a single train-test split. Cross-validation would provide a more robust assessment of model performance.

### Temporal Validation

Customer behaviour can change over time. A time-based validation approach could provide a more realistic assessment of how the model performs on future observations.

### Model Interpretability

Random Forest provided stronger predictive performance but is less directly interpretable than Logistic Regression. Explainable AI methods such as SHAP could be explored in future work.

### Classification Threshold

Different classification thresholds could change the balance between precision and recall. Threshold optimisation could be investigated depending on the business cost of false positives and false negatives.

### Additional Data

Additional information such as customer service interactions, complaints, payment behaviour and historical changes in transaction activity could potentially improve future churn analysis.

## Visualisations

The project includes multiple visualisation types to communicate analytical findings, including:

* Bar charts for comparing attrition rates across customer groups
* Feature importance visualisation
* Correlation heatmap
* Confusion matrix
* ROC curve

These visualisations support different business and analytical questions and help communicate both exploratory and predictive findings.

## Technologies Used

* Python
* pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* scikit-learn
* Jupyter Notebook
* GitHub

## Project Structure

```text
Customer-Churn-Analysis-and-Prediction/
│
├── BankChurners.csv
├── customer_churn_cleaned.csv
├── Customer_Churn_Analysis_and_Prediction.ipynb
├── logistic_regression_coefficients.csv
├── model_comparison_results.csv
└── README.md
```

## How to Run the Project

### 1. Clone or download the repository

Download the project files from GitHub.

### 2. Install the required Python libraries

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

### 3. Open the notebook

Launch Jupyter Notebook or JupyterLab and open:

```text
Customer_Churn_Analysis_and_Prediction.ipynb
```

### 4. Run the notebook

Run the notebook cells in order, starting with the data loading and preparation stages.

The original dataset should be available in the same project directory if the notebook is being run from the beginning.

## Future Improvements

Future development could include:

* Cross-validation and additional model validation.
* Testing additional classification algorithms.
* Explainable AI using SHAP or similar techniques.
* Churn probability analysis.
* Interactive dashboard development.
* Time-based model validation.
* Model threshold optimisation.
* Monitoring model performance using new customer data.
* Development of a more complete customer retention decision-support prototype.

## Learning Reflection

This project provided practical experience in applying the data analytics workflow to a real-world business problem.

It strengthened my skills in:

* Data cleaning and preparation
* ETL
* Exploratory data analysis
* Statistical hypothesis testing
* Data visualisation
* Feature analysis
* Supervised machine learning
* Model evaluation
* Business interpretation of analytical results

The project also improved my understanding of the importance of avoiding data leakage, considering class imbalance, selecting appropriate evaluation metrics and connecting technical findings to real-world business requirements.

## Author

**Yetunde Okunlade**

Level 3 Diploma in Data Analytics with AI
