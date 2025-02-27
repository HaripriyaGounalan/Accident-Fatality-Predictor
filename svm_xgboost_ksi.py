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
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

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

    model_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('svm', SVC(random_state=17))
    ])

    preprocessor1 = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder='passthrough')

    model_xgb = Pipeline([
        ('preprocessor', preprocessor1),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=17))
    ])

    # RBF Kernel: C=100, gamma=0.03, Poly Kernel: C=0.1, degree=3, gamma=3.0
    param_grid_svm = [
        {'svm__kernel': ['linear'], 'svm__C': [1]},  # linear kernel    
        {'svm__kernel': ['rbf'], 'svm__C': [100], 'svm__gamma': [0.03]},  # rbf kernel
        {'svm__kernel': ['poly'], 'svm__C': [0.1], 'svm__gamma': [3.0], 'svm__degree': [3,4]}  # poly kernel 
    ]

    param_grid_xgb = {
    'xgb__learning_rate': [0.2],
    'xgb__max_depth': [7],
    'xgb__n_estimators': [200],
    'xgb__subsample': [0.7],
    'xgb__colsample_bytree': [0.7]
    }

    # param_grid_combined = {
    # 'svm__svm__kernel': ['linear', 'rbf', 'poly'],
    # 'svm__svm__C': [0.1, 1, 100],
    # 'svm__svm__gamma': [0.03, 3.0],
    # 'svm__svm__degree': [3, 4],
    # 'xg_boost__xgb__learning_rate': [0.2],
    # 'xg_boost__xgb__max_depth': [7],
    # 'xg_boost__xgb__n_estimators': [200],
    # 'xg_boost__xgb__subsample': [0.7],
    # 'xg_boost__xgb__colsample_bytree': [0.7]
    # }
    
    param_grid_combined = {
    'svm__svm__kernel': ['rbf'],
    'svm__svm__C': [100],
    'svm__svm__gamma': [0.03],
    'svm__svm__degree': [3],
    'xg_boost__xgb__learning_rate': [0.2,0.25],
    'xg_boost__xgb__max_depth': [8],
    'xg_boost__xgb__n_estimators': [200,250],
    'xg_boost__xgb__subsample': [0.7,0.75,0.8],
    'xg_boost__xgb__colsample_bytree': [0.7,0.75,0.8]
    }
    

    # Voting Classifier (Hard Voting)
    voting_clf = VotingClassifier(estimators=[('svm', model_svm), ('xg_boost', model_xgb)], voting='hard')

    # Step c: Train the model with GridSearchCV
    grid_search_ksi = GridSearchCV(estimator=voting_clf, param_grid=param_grid_combined, scoring='accuracy', refit=True, verbose=3)

    # Fit the model
    grid_search_ksi.fit(X_train, y_train)

    best_model = grid_search_ksi.best_estimator_

    # Save the best model
    print("\nBest Parameters:", grid_search_ksi.best_params_)
    print("Best Estimator:", grid_search_ksi.best_estimator_)
    print("Best Score:", grid_search_ksi.best_score_)

    # Evaluate the model
    accuracy = best_model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred in Pre-processing & Training the model: {e}")