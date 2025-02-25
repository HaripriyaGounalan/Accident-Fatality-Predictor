import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib

# Setting a random seed for consistency
np.random.seed(17)

# Step a: Load & Check the data
try:
    # Load the data
    data_ksi = pd.read_csv('./Total_KSI.csv')

    if data_ksi is not None:
        pd.set_option("display.max_columns", 100)
        # Initial Investigations
        print("\nDisplaying First 3 Records:\n", data_ksi.head(3))
        print("\nShape of the dataframe:", data_ksi.shape)
        print("\nCheck the type of the dataframe:", type(data_ksi))
        print("\nData Description:\n", data_ksi.describe())
        print("\nColumn Information:")
        data_ksi.info()
        print("\nMissing Values Per Column:\n", data_ksi.isnull().sum())

        # Drop unnecessary columns
        data_ksi = data_ksi.drop(columns=['INDEX', 'ACCNUM', 'OBJECTID', 'HOOD_158', 'NEIGHBOURHOOD_158', 
                                          'HOOD_140', 'NEIGHBOURHOOD_140', 'STREET1', 'STREET2', 
                                          'OFFSET', 'FATAL_NO', 'DISTRICT', 'DIVISION'])

        # Separate the features & target
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(data_ksi["ACCLASS"])  
        features = data_ksi.drop(columns=["ACCLASS"])

        print("road class", features["ROAD_CLASS"].unique())
        print("\nFeatures:\n", features.info())

        # Encode the target variable (ACCLASS) using LabelEncoder
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)

        # Split the data into train & test
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=17)

except Exception as e:
    print(f"An error occurred in Loading & Checking Data: {e}")

# Step b: Pre-process and train the model
try:
    # Identify numerical and categorical features
    num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = features.select_dtypes(include=['object']).columns.tolist()

    print("\nCategorical Features:", cat_features)

    print("\nMissing Values in Dataset (after cleaning):")
    print(features.isnull().sum())

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder='passthrough')

    # Define XGBoost model pipeline
    pipe_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=17))
    ])

    # Define hyperparameter grid for XGBoost
    param_grid_xgb = {
        'xgb__n_estimators': [50, 100, 200], 
        'xgb__max_depth': [3, 5, 7],  
        'xgb__learning_rate': [0.01, 0.1, 0.2],  
        'xgb__subsample': [0.7, 1.0],  
        'xgb__colsample_bytree': [0.7, 1.0]  
    }

    grid_search_xgb = GridSearchCV(estimator=pipe_xgb, param_grid=param_grid_xgb, 
                                   scoring='accuracy', refit=True, verbose=3, cv=5)

    # Step c: Train the model
    grid_search_xgb.fit(X_train, y_train)

    best_model = grid_search_xgb.best_estimator_

    # Save the best model
    print("\nBest Parameters:", grid_search_xgb.best_params_)
    print("Best Estimator:", grid_search_xgb.best_estimator_)
    print("Best Score:", grid_search_xgb.best_score_)

    accuracy = best_model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred in Pre-processing & Training the model: {e}")
