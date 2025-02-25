import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.ensemble import RandomForestClassifier

# Setting a random seed at beginning for consistency
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
        data_ksi = data_ksi.drop(columns=['INDEX', 'ACCNUM', 'OBJECTID', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'STREET1', 'STREET2', 'OFFSET', 'FATAL_NO', 'DISTRICT', 'DIVISION'])

        # Separate the features & target
        target = data_ksi["ACCLASS"]
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

    print("\nNumerical Features:", num_features)
    print("\nCategorical Features:", cat_features)

    print("\nMissing Values in Dataset (after cleaning):")
    print(features.isnull().sum())

    # Handling missing values and transformations
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  
        ('scaler', StandardScaler())  
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('encoder', OneHotEncoder(handle_unknown="ignore"))  
    ])

    # Combine transformations using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])


    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=17))
    ])

    # GridSearchCV can be used for hyperparameter tuning
    param_grid_rf = {
        'rf__n_estimators': [50, 100, 200],  
        'rf__max_depth': [None, 10, 20, 30],  
        'rf__min_samples_split': [2, 5, 10],  
        'rf__min_samples_leaf': [1, 2, 4],  
        'rf__bootstrap': [True, False]  
    }

    grid_search_rf = GridSearchCV(estimator=pipe_rf, param_grid=param_grid_rf, scoring='accuracy', refit=True, verbose=3)

    # Step c: Train the model
    grid_search_rf.fit(X_train, y_train)

    best_model = grid_search_rf.best_estimator_

    # Save the best model
    print("\nBest Parameters:", grid_search_rf.best_params_)
    print("Best Estimator:", grid_search_rf.best_estimator_)
    print("Best Score:", grid_search_rf.best_score_)

    accuracy = best_model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred in Pre-processing & Training the model: {e}")