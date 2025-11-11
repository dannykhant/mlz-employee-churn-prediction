# Project Overview: Employee Churn Prediction

### Disclaimer

This project is intended to fulfill the requirements of the Machine Learning Zoomcamp midterm project.

## Problem Description

Employee turnover or "churn" is a major challenge for organizations. High churn rates increase recruitment and training costs, disrupt team productivity, and can harm morale.
This project aims to predict whether an employee is likely to leave the company based on their demographic, performance, and compensation data.

By building a predictive model, the HR department can identify at-risk employees early and take proactive measures to improve retention (e.g., career growth opportunities, workload balance, or salary adjustments).

### Who Benefits

- Human Resources (HR) teams: Gain insights to make data-driven retention strategies.

- Management: Reduce turnover-related costs and maintain workforce stability.

- Employees: Benefit from improved engagement and retention initiatives.

### How the Model Will Be Used

The final model will take as input employee data (e.g., salary level, age, work experiences) and output the probability that an employee will leave.
This probability can be integrated into an HR dashboard or used in batch processing to flag employees for review.

### Evaluation Metric

Since this is a classification problem and the dataset may not be perfectly balanced, the following metrics will be used:

- Accuracy: Measures overall correctness of predictions.

- ROC-AUC Score: Evaluates how well the model distinguishes between churned and non-churned employees across all thresholds.

- Precision and Recall: Provide insights into false positives and false negatives, respectively.

The ROC-AUC score will serve as the primary evaluation metric, as it offers a robust measure of separability between classes, even under moderate imbalance.

### Why This Problem Matters

Employee retention directly impacts a company’s financial health and culture.
Replacing an employee can cost 30–200% of their annual salary, depending on the role.
Predictive analytics can empower organizations to act before churn happens, making workforce management more strategic and humane.

## EDA

#### Null Checking
![Null Checking](images/eda_null_check.png)
Observation: The dataset contains no missing or null values.

#### Categorical Unique Counts
![Categorical Unique Counts](images/eda_unique_counts.png)
Observation: All categorical features contain between 2 and 3 distinct categories.

#### Numerical Description
![Numerical Description](images/eda_numerical.png)
Observation: The average employee age is around 29, indicating that most individuals fall within the mid-career range. Both experience and job tenure have a minimum value of 0, representing employees whose duration has not yet reached one full year.

#### Mutual Information on Categorical
![MI](images/eda_mi.png)
Observation: The payment tier shows the highest mutual information score, indicating that employee churn is strongly influenced by salary level.

#### Correlation on Numerical
![Coor](images/eda_corr.png)
Observation: Job tenure shows a highest correlation with churn, and all numerical features exhibit negative correlations, meaning that as these values increase, the likelihood of churn decreases.

*Detailed information is available in the notebook.ipynb file.*

## Model Training

Four different models were trained: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Fine-tuning has also been performed for each model to optimize their performance.

#### Model Selection
![Models](images/model_comparison.png)
Observation: As shown in the figure above, the ROC-AUC scores of Random Forest and Gradient Boosting are quite similar (both around 88%). Therefore, Gradient Boosting is selected as the final model.

#### Cross Validation
![Cross Validaton](images/model_cross_val.png)
Observation: Based on the cross-validation results, the model demonstrates satisfactory and consistent performance, making it suitable as our final model.

*Detailed information is available in the notebook.ipynb file.*

## Dependencies
The necessary dependencies are listed below.

 - fastapi[standard]
 - jupyter
 - pandas
 - requests
 - scikit-learn
 - seaborn
 - xgboost

*You can install the dependencies using any Python package manager, such as pip or pipenv, or follow the instructions provided below.*

### Installation Guide
> `uv` must be installed first in order to perform the synchronization.

Mac & Linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> Execute the following command to install the dependencies.

```
uv sync
```

> Activate the virtual environment using the following command.
```
source .venv/bin/activate
```

## Scripts
#### `train.py` 
Trains the model using the prepared dataset and saves the trained model artifact for later use.
```
python train.py
```

#### `predict.py`
Loads the saved model and performs inference on new or unseen data.
```
python predict.py
```

#### `serve.py`
Hosts the trained model as a web service, allowing users to make predictions through an API or web interface.
```
fastapi dev serve.py
```
*Ensure that you have installed the dependency `fastapi[standard]` before running this command.*


## Reproducibility
After installing all required dependencies, you can reproduce the results by running the notebook (notebook.ipynb) or executing any of the scripts (train.py, predict.py, serve.py) as per the provided instructions.

## Model Deployment
```
fastapi run serve.py
```

## Containerization
