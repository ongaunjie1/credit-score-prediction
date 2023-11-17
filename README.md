# credit-score-prediction
Link to streamlit app: https://credit-score-prediction.streamlit.app/ 
* **NOTE**: The app might run out of resources if too many predictions are made, it will require me to reboot the app. Let me know if it happens!)

## About Dataset
| Field Name    | Description                                                    | Type    |
|---------------|----------------------------------------------------------------|---------|
| age           | Age of the individual.                                         | int64   |
| occupation    | Occupation of the individual.                                  | Object  |
| annual_income | Annual income of the individual.                               | float64 |
| num_bank_acc  | Number of bank accounts owned.                                 | int64   |
| num_credit_card| Number of credit cards owned.                                  | int64   |
| interest_rate | Interest rate of credit card.                                  | float64   |
| delay_from_due_date | Delayed days from payment's due date.                      | int64   |
| outstanding_debt | Amount of outstanding debt.                                | float64 |
| credit_util_ratio | Credit utilization ratio.                                  | float64 |
| credit_history_age | Credit history age.                                      | float64 |
| payment_of_min_amount | Indicates if the minimum amount is paid.                 | Object  |
| installment_per_month | Monthly installment amount.                              | float64 |
| amount_invested_monthly | Amount invested monthly.                               | float64 |
| payment_behavior | Payment behavior category.                                | Object  |
| monthly_balance | Monthly balance                                           | float64 |
| credit_score  | Credit score.                                                | int64   |

## Goal of the project
* Train a multi-class classification model to predict the credit score of an individual
* Target variable: **credit_score**
* The target variable contains 3 classes = ['Poor', 'Standard, 'Good']

# Refer below for the all the steps taken:
## Step 1: Data cleaning and preprocessing of the raw-data
* The raw data is cleaned and process within the data-cleaning notebooks (train and test notebooks were cleaned the same way, there are separate files due to the long cleaning process)
* Refer to the data-cleaning notebooks for more informations
### a TLDR on the data-cleaning.ipynb:
* Cleaning dirty data (ie. removing invalid data, symbols within texts and etc)
![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/8610636f-6d49-4fdc-8f37-0169981ca83b)
* Preliminary feature dropping ,removing features that are completely irrelevant
* Reviewing data's outlier and removing outliers outside of the Interquartile Range (IQR) range
* Mapping target variable's 3 classes into numeric format ('0: Poor', '1: Standard', '2: Good')
* Standardization of data types before EDA and modelling

## Step 2: Reading in the cleaned/preprocessed data and performing EDA (Exploratory Data Analysis)

## a) Analyze distribution of age

![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/b0bb7fda-3a9b-477d-b4bb-c1e81e470ef6)

## b) Explore Credit score distribution by occupation

![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/8158cb72-0b6f-4ced-93a8-6f02a450c9ac)

## c) Explore credit score distribution by payment_of_min_amount
  
![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/8279c025-7b87-4eaa-a896-1b5de17a4ffd)

## d) Explore credit score distribution by payment behavior
  
![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/f31b220f-36e4-4081-b31a-ae0162170f2d)

## e) Explore target variable distribution

![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/cdfa6cc8-62c2-46a3-8aaa-48018f60e272)

## f) Analyze the relationship between features and target variables using boxplot

![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/b8db7ada-050b-46f5-bf5e-1149cabd1d13)

### Insights from EDA:
* a) Distribution of age is right skewed
* b) The distribution of credit score by occupation is evenly distributed (will be dropped in feature selection)
* c) There is a trend where individuals who pays credit payment in minimum will have a lower credit score
* d) Payment behavior of "low spending with small payments" have a higher concentration of individuals who have poor/standard credit scores
* e) The target variable is imbalanced. Options are use resampling techniques ,utilize models that can handle imbalanced dataset like Random Forest and boosting algorithms or apply weights to the minority class. For this project, random forest and xgboost will be used to handle imbalanced dataset.
* f) Based the box plots, annual_income, credit_util_ratio, amount_invested_monthly seems to have no impact on credit score (this will be verified again using feature correlation analysis below)

## Step 3: Data Preparation and Feature engineering
* Performing one-hot-encoding on the categorical features
* Plot a heatmap using seaborn for feature correlation analysis
![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/d9699195-af48-47d6-adee-6dcefcb7f054)

### Insights from feature correlation analysis:
* There are a few features that are moderately correlated with the target variable, none of them with high correlations ( > 0.7). As for correlation between input features, none are redundant features.
* Note: There are also features that are inversely correlated with the target variable (ie. outstanding debt has a correlation of 0.41, this means that as debt decreases, credit score increases which is reasonable for credit score)
* Also, it shows that monthly_balance, amount_invested_monthly, installment_per_month are positively correlated with annual income. This makes sense because as income increases, these 3 features will also increases. 
* To summarize, the features with <= 0.5 will be removed (absolute value). These features are credit_util_ratio, amount_invested_monthly, payment_behavior, occupation.

### Feature input for training 
| age | annual_income | num_bank_acc | num_credit_card | interest_rate | delay_from_due_date | outstanding_debt | credit_history_age | installment_per_month | monthly_balance | payment_of_min_amount_Yes |
|----|---------------|--------------|-----------------|----------------|----------------------|-------------------|---------------------|----------------------|------------------|--------------------------|
| 23 | 19114.12      | 3            | 4               | 3              | 3                    | 809.98            | 22.1                | 49.57                | 312.49           | False                    |
| 23 | 19114.12      | 3            | 4               | 3              | 5                    | 809.98            | 22.4                | 49.57                | 223.45           | False                    |
| 23 | 19114.12      | 3            | 4               | 3              | 6                    | 809.98            | 22.5                | 49.57                | 341.49           | False                    |
| 23 | 19114.12      | 3            | 4               | 3              | 3                    | 809.98            | 22.7                | 49.57                | 244.57           | False                    |
| 28 | 34847.84      | 2            | 4               | 6              | 7                    | 605.03            | 26.8                | 18.82                | 484.59           | False                    |

## Step 4: Model Training
* Splitting dataset using train_test_split **(80% training dataset, 20% test dataset)**

## Step 6: Model training
### 1st Model: Using XGBoost Regressor
### Result:
  
| Metric | Value                |
|--------|----------------------|
| MSE    |    1,939,965,795     |
| MAE    |       28,216         |
| R2     |     **0.790**       | 

### 2nd Model: Using Random Forest Regressor

| Metric | Value                |
|--------|----------------------|
| MSE    |    2,047,237,885     |
| MAE    |       28,953         |
| R2     |     **0.778**        | 

### 3rd Model: Using Linear Regression

| Metric | Value                |
|--------|----------------------|
| MSE    |    2,960,230,826     |
| MAE    |       34,375         |
| R2     |     **0.679**        | 

## Step 7: Model Selection and fine-tuning using GridSearchCV
* From the preliminary model fitting, it showed that XGboost performed the best among the three models with a R^2 of **0.790**. Hence, it is selected for fine-tuning.
### Hyperparamter grid
```
param_grid = {
    "learning_rate": [0.05, 0.10, 0.15],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2],
    "colsample_bytree": [0.3, 0.4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=0, n_jobs=-1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)
```
### Fine-tuned model: Using XGboost Regressor

| Metric | Value                |
|--------|----------------------|
| MSE    |    1,785,590,381     |
| MAE    |       27,957         |
| R2     |     **0.806**        | 

* The model improved slightly compared to the base-model.

### Possible improvement for the model
* Revisit feature engineering
* Normalizing Variables
* Regularization
* Further expand the hyperparameters grid to improve performance

## Step 8: Creating a streamlit-app using the fine-tuned model
* The app will be capable of taking in end user's input of the features and predicting the sale price
* Link to the app: https://real-estate-price-prediction-app.streamlit.app/
* Feel free to test the app ! Let me know if you encountered any issues with the app.
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/1ca8daae-ed40-4025-8317-88fc5c235ba6)





  




Improvement to the model
Use resampling techniques to balance the target variable









  








