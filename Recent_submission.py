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

data = pd.read_parquet("/kaggle/input/msdb-2024/train.parquet")
test_data = pd.read_parquet("/kaggle/input/msdb-2024/final_test.parquet")
df_ext = pd.read_csv("/kaggle/input/msdb-2024/external_data.csv", parse_dates=["date"])

threshold = 0.4  # Set the threshold
df_ext = df_ext.loc[:, df_ext.isnull().mean() <= threshold]

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

# Define objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    # Create pipeline with One-hot Encoding (compatible with HistGradientBoosting)
    pipeline = Pipeline(
        steps=[
            ("vectorizer", TableVectorizer()),
            (
                "regressor",
                XGBRegressor(
                    **params, random_state=42
                ),
            ),
        ]
    )

    # Cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean()


# Run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=3600)


# Get the best hyperparameters and RMSE
print("Best RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)

best_params = study.best_params
final_model = Pipeline(
    steps=[
        ("vectorizer", TableVectorizer()),
        (
            "regressor",
                XGBRegressor(
                    **best_params, random_state=42
            ),
        ),
    ]
)
final_model.fit(X_train, y_train)

test_data_predictions = final_model.predict(test_data_with_ext[features])
# Add predictions to the test dataset

test_data_with_ext['log_bike_count'] = test_data_predictions

predictions_output = test_data_with_ext[['log_bike_count']].copy()

predictions_output.insert(0, 'Id', predictions_output.index)

# Save to CSV file

output_file_path = 'submission.csv'

predictions_output.to_csv(output_file_path, index=False)
