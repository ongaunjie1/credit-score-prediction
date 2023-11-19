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
### Overview on the data-cleaning.ipynb:
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
* Selecting input features and dropping insignificant features such as credit_util_ratio, amount_invested_monthly, payment_behavior, occupation.
* Creating features X and target variable y
* Splitting dataset using train_test_split **(80% training dataset, 20% test dataset, stratify=y (for imbalance target variable))**
* The data will be trained on 3 different models: Decision Tree Classifier, Random Forest Classifier and XGboost Classifier.
* You may refer to the model.ipynb notebook for information on the training details

## Model Performance (Using base-model, without fine-tuning)
| Model                      | Accuracy | Precision | Recall   | F1 Score |
|----------------------------|----------|-----------|----------|----------|
| Random Forest Classifier   | 0.7951   | 0.7951    | 0.7951   | 0.7944   |
| XGBoost Classifier         | 0.7755   | 0.7748    | 0.7755   | 0.7744   |
| Decision Tree Classifier   | 0.7238   | 0.7239    | 0.7238   | 0.7238   |
  
## Step 5: Model fine-tuning for XGboost
* Although Random Forest Classifier performed the best in accuracy. XGboost was selected for fine-tuning.
* This is because, the fine-tuned model will be deployed to the streamlit community cloud which requires model size to be <= 25mb.
* Hence, the model selected was XGboost due to its light-weight capability.
* Alternatively, we can upload bigger models with higher accuracy using Git LFS but that requires a fee for a bigger storage and bandwidth.
* The model was fine-tuned with RandomizedSearchCV instead of GridSearchCV for a more efficient and computationally lighter optimization process.

### Tuning Params grid used for fine-tuning XGboost model
```
param_dist = {
    'n_estimators': randint(100, 500),  # Adjust the range based on your problem
    'learning_rate': [0.01, 0.1],
    'max_depth': randint(1, 20),
    'subsample': [ 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'gamma': [0, 1, 2, 3],
    'min_child_weight': [ 2, 3, 4],
}
```
### Best params from fine-tuning:
* Best Hyperparameters: {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 17, 'min_child_weight': 4, 'n_estimators': 361, 'subsample': 0.9}

### Model Performance using the best hyperparameters from the fine-tuning process
| Model                              | Accuracy | Precision | Recall  | F1 Score |
| ---------------------------------- | -------- | --------- | ------- | -------- |
| XGBoost Classifier (randomized search) | 0.79295  | 0.792955  | 0.79295 | 0.791597 |
* It is an improvement from the base model but unfortunately its size is > 25 mbm.
* Trial and error needed to achieve a smaller size.

### Trial and error the best hyperparameters to achieve a smaller weight for the model, refer below for the parameters used to achieve <25mb 
```
xgb_classifier_test = XGBClassifier(
    n_estimators=150,  
    learning_rate=0.01,
    max_depth=17, 
    subsample=0.9, 
    colsample_bytree=0.7,  
    gamma=0,
    min_child_weight=4,
)
```
### Final model's results:
| Model                              | Accuracy | Precision | Recall  | F1 Score |
| ---------------------------------- | -------- | --------- | ------- | -------- |
| XGBoost Classifier (randomized search: smaller version) | 0.786197  | 0.785945	  | 0.786197 | 0.784046 |
* Slightly worse performance but it is light-weight and ready to be deployed to streamlit community cloud

### Confusion Matrix Plot for the final selected model:
![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/bb41d940-e9e5-44a9-ba28-0f99325e99b4)

### Feature importance plot:
![image](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/2c918f54-8e86-48ae-a3d2-51add3308eaa)

### Possible improvement for the model
* Could try resampling technique or applying weights to minority class for the imbalanced target variable
* Normalizing numeric variables using scaler
* For local inference, could try and fine-tune the random forest classifier for a better model performance

## Step 5: Creating a streamlit-app using the fine-tuned model
* The app will demonstrate the model by predicting an individual's credit score based on user input features.
* Link to the app: https://credit-score-prediction.streamlit.app/
* Feel free to test the app ! Let me know if you encountered any issues with the app.

## How to use the app ?
* The left side of the app will be the user's input for the features
* Once selected your features, you may click on the predict button
* The predictions will be shown on the right as texts
* You can refer to the predicted_test.csv file within this github repo for a sample on what values to use. The csv file is a separate test dataset predicted using the XGBoost Classifier (randomized search: smaller version) model.
* You could also refer to the heat-map used for correlation analysis to understand further on what affects a person's credit score.

## App prediction examples:
### A Good credit score prediction will be shown as a green text:
![good-credit-prediction](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/688add75-e770-46c8-9db5-df8276cc9207)
### A Standard credit score will be shown as a yellow text:
![average-credit-prediction](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/30e18b71-e6b2-4641-8732-7d78c05eaf39)
### A Poor credit score will be shown as a red text:
![poor-credit-prediction](https://github.com/ongaunjie1/credit-score-prediction/assets/118142884/5f4aabf6-ffdb-4761-a98c-c4a4306d6c3c)


