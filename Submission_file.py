import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# Creating a pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from skrub import TableVectorizer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def _merge_external_data(X, df_ext):
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    df_ext["date"] = pd.to_datetime(df_ext['date']).astype('datetime64[us]')
    
    X["orig_index"] = np.arange(X.shape[0])
    # X = pd.merge_asof(
    #     X.sort_values("date"), df_ext[["date", "t", "td", "ww", "u"]].sort_values("date"), on="date"
    # )
    X = pd.merge_asof(
        X.sort_values("date"), df_ext.sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

data = pd.read_parquet("data/train.parquet")
test_data = pd.read_parquet("data/final_test.parquet")
df_ext = pd.read_csv("external_data/external_data.csv", parse_dates=["date"])

train_data_with_ext = _merge_external_data(data, df_ext)
train_data_with_ext = train_data_with_ext.reset_index(drop=True)

test_data_with_ext = _merge_external_data(test_data, df_ext)
test_data_with_ext = test_data_with_ext.reset_index(drop=True)

# Parse the 'date' column and extract time-based features for the training data

train_data_with_ext['date'] = pd.to_datetime(train_data_with_ext['date'])

train_data_with_ext['hour'] = train_data_with_ext['date'].dt.hour

train_data_with_ext['day_of_week'] = train_data_with_ext['date'].dt.dayofweek

train_data_with_ext['day'] = train_data_with_ext['date'].dt.day

train_data_with_ext['month'] = train_data_with_ext['date'].dt.month

train_data_with_ext['year'] = train_data_with_ext["date"].dt.year

# Parse the 'date' column and extract time-based features for the test data

test_data_with_ext['date'] = pd.to_datetime(test_data_with_ext['date'])

test_data_with_ext['hour'] = test_data_with_ext['date'].dt.hour

test_data_with_ext['day_of_week'] = test_data_with_ext['date'].dt.dayofweek

test_data_with_ext['day'] = test_data_with_ext['date'].dt.day

test_data_with_ext['month'] = test_data_with_ext['date'].dt.month

test_data_with_ext['year'] = test_data_with_ext["date"].dt.year

from sklearn.model_selection import train_test_split



# Define features and target
# external_cols = list(df_ext.columns).remove('numer_sta').remove('date')

features = [
    col for col in (
        ['counter_name', 'site_name', 'hour', 'day_of_week', 'month', 'year', 'latitude', 'longitude', 'day'] +
        list(df_ext.columns)
    ) if col not in ['numer_sta', 'date']
]
target = 'log_bike_count'



# Split the data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(

    train_data_with_ext[features], train_data_with_ext[target], test_size=0.2, random_state=42

)



# Check the shapes of the resulting datasets

X_train.shape, X_val.shape, y_train.shape, y_val.shape

# Defining the pipeline
pipeline = Pipeline(
    steps=[
        ("vectorizer", TableVectorizer()),
        ("regressor", XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.2, random_state=42, colsample_bytree=0.8, subsample=1.0)),
    ]
)

# Fit the pipeline on the data
pipeline.fit(X_train, y_train)

y_val_pred_hgb = pipeline.predict(X_val)

test_data_predictions = pipeline.predict(test_data_with_ext[features])
# Add predictions to the test dataset

test_data_with_ext['log_bike_count'] = test_data_predictions

predictions_output = test_data_with_ext[['log_bike_count']].copy()

predictions_output.insert(0, 'Id', predictions_output.index)

# Save to CSV file

output_file_path = 'data/Test_Predictions_xgb_ext_skrub2_python_file.csv'

predictions_output.to_csv(output_file_path, index=False)
