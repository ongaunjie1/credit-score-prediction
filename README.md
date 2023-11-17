# credit-score-prediction
Link to streamlit app: https://credit-score-prediction.streamlit.app/

## About Dataset
| Field Name    | Description                                                    | Type  |
|---------------|----------------------------------------------------------------|-------|
| PropertyID    | A unique identifier for each property.                         |  int64 |
| PropType      | The type of property (e.g., Commercial or Residential).        | Object |
| taxkey        | The tax key associated with the property.                      | Object |
| Address       | The address of the property.                                   | Object |
| CondoProject  | Information about whether the property is part of a condominium project (NaN indicates missing data). | Object  |
| District      | The district number for the property.                          | Object |
| nbhd          | The neighborhood number for the property.                      | Object |
| Style         | The architectural style of the property.                       | Object |
| Extwall       | The type of exterior wall material used.                       | Object |
| Stories       | The number of stories in the building.                         | float64 |
| Year_Built    | The year the property was built.                               |  int64 |
| Rooms         | The number of rooms in the property.                           | Object |
| FinishedSqft  | The total square footage of finished space in the property.    |  int64 |
| Units         | The number of units in the property (e.g., apartments in a multifamily building). | Object |
| Bdrms         | The number of bedrooms in the property.                        | Object |
| Fbath         | The number of full bathrooms in the property.                  | Object |
| Hbath         | The number of half bathrooms in the property.                  | Object |
| Lotsize       | The size of the lot associated with the property.              |  int64  |
| Sale_date     | The date when the property was sold.                           | datetime |
| Sale_price    | The sale price of the property.                                |  int64 |

## Goal of the project
* To train a regression model to predict the sales price of a real estate property based on different features.
* Target variable: Sale_price (int64).

# Refer below for the all the steps taken:
## Step 1: Data merging
* From the source, the datasets are separated by year ranging from 2002-2022. Hence, concatenation is required.
* Refer to the file concat.ipynb for more information on the data merging process.
### a TLDR on the concat.ipynb:
* Changing Sale_date format into YYYY-MM for the 2019-2022 dataset to align with the date format of the 2002-2018 datasets
* Remove missing values (NaN values)
* Dropping insignificant features
* Standardized the data types before merging into a single file

## Step 2: Reading in the merged file and performing feature engineering
* Skipped data cleaning in this step as the process was performed in the concat.ipynb
* Created new features: year_sold and month_sold

## Step 3: EDA (Exploratory Data Analysis)
## a) Box-plot visualization (to identify outliers and dropping extreme outliers)
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/b75d11f5-cc25-48ef-9892-3cddaae9aece)

## b) Analyze distribution of target variable (Sale_price)

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/c8afe42c-7fd0-492d-a78e-7d527ee66df6)

## c) Analyze the relationship between Fin_sqrt and Sale_price
  
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/311d06a7-9e1f-4fab-a5e0-b38e44be96e1)

## d) Analyze relationship between lotsize and sale price
  
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/0a5eea3b-31e7-42c0-80aa-a1912620e46d)

## e) Analyze average sale price over the years

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/b8b2db47-9da4-482d-9281-9c16676c631c)

## f) Analyze average sale price by year built

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/00172e31-c801-4e6a-83c0-b0bc59b3c66a)

## g) Analyze average sale price by number of rooms

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/bf700c62-9907-457e-ba95-1900c994fbf4)

## h) Analyze average sale price by number of bedrooms

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/9f9df120-dffd-4498-b439-022a6a45f767)

## i) Analyze average sale price by number of stories
  
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/a897747a-1e5f-45b8-afbc-982ec631b20a)

## j) Analyze average sale price by number of full bathrooms
  
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/44913deb-ab32-492b-9664-c4ec04d59e1f)

## k) Analyze average sale price by district

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/b21fe7d7-6bf0-4d36-b291-7f6f91c393e8)

## l) Analyze average sale price by style

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/7ffc260c-0ff3-4e5d-9541-fb498e6193c1)

## m) Analyze average sale price by extwall

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/b4c80dcf-074b-4447-8a82-fe6c8ca71f8b)

## n) Analyze average sale price by month

![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/157be06e-b6b8-4eac-85f4-c9b9fb99041a)

## EDA TLDR:
### From the EDA, features that could impact the sale price of a real estate property are:
* Finished Fin_sqft
* Lot size
* Year Sold
* Year Built
* Number of Rooms
* Number of BedRooms
* Number of Stories
* Number of fbath
* District Type
* Style Type
* Extwall type
### Features that seems insignificant are:
* Month
* Address
* Nbhd
* PropType (Due to huge imbalance)
* Units

## Step 4: Data pre-processing
* Dropping insignificant features: ['Address', 'PropType', 'Nbhd','month_sold','Hbath', 'Units']
* Converting categorical features into the correct datatype (object) before one-hot-encoding
* Perform One-hot-encoding for categorical features: ['District', 'Style', 'Extwall', 'Nr_of_rms', 'Bdrms', 'Fbath']

| Stories | Year_Built | Fin_sqft | Lotsize | year_sold | Sale_price | District_1 | District_2 | District_3 | District_4 | ... | Bdrms_32 | Fbath_0 | Fbath_1 | Fbath_2 | Fbath_3 | Fbath_4 | Fbath_5 | Fbath_6 | Fbath_7 | Fbath_10 |
| ------- | ---------- | -------- | ------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- | --- | -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- |
| 2.0     | 1913       | 3476     | 5040    | 2002      | 42000      | False      | False      | False      | False      | ... | False    | False   | True    | False   | False   | False   | False   | False   | False   | False    |
| 2.0     | 1897       | 1992     | 2880    | 2002      | 145000     | False      | False      | True       | False      | ... | False    | False   | False   | True    | False   | False   | False   | False   | False   | False    |
| 2.0     | 1907       | 2339     | 3185    | 2002      | 30000      | False      | False      | False      | True       | ... | False    | False   | True    | False   | False   | False   | False   | False   | False   | False    |
| 2.0     | 1890       | 2329     | 5781    | 2002      | 66500      | False      | False      | False      | True       | ... | False    | False   | True    | False   | False   | False   | False   | False   | False   | False    |
| 2.5     | 1891       | 7450     | 15600   | 2002      | 150500     | False      | False      | False      | True       | ... | False    | False   | False   | False   | False   | False   | False   | True    | False   | False    |

## Understanding correlation between independent variables using heatmap
![image](https://github.com/ongaunjie1/Real-estate-price-prediction/assets/118142884/aed1184d-9a49-4f1f-be12-dae52ff416b7)

## Step 5: Data Preparation (Split dataset) before model training
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





  











  








