import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def prepare_features(df, feature_columns):
    # Separar características y etiquetas
    X = df[feature_columns]
    y = df['cardio']
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X_scaled, y, test_size=0.2, random_seed=42):
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_seed)

def transform_features(X_train, X_test, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    return X_train_poly, X_test_poly, poly