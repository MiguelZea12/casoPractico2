import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

def train_model(X_train_poly, y_train):
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model

def evaluate_model(model, X_train_poly, X_test_poly, y_train, y_test):
    # Predicciones
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calcular métricas
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

def optimize_model(model, X_train_poly, y_train):
    def objective_function(params):
        model.coef_ = params[:-1]
        model.intercept_ = params[-1]
        preds = model.predict(X_train_poly)
        return mean_squared_error(y_train, preds)

    # Inicialización de parámetros
    initial_params = np.append(model.coef_, model.intercept_)
    result = minimize(objective_function, initial_params, method='BFGS')

    # Actualizar parámetros del modelo
    model.coef_ = result.x[:-1]
    model.intercept_ = result.x[-1]
    
    return model

def identify_risk_factors(model, poly, X):
    import pandas as pd
    
    factor_importance = pd.Series(
        np.abs(model.coef_), 
        index=poly.get_feature_names_out(input_features=X.columns)
    )
    return factor_importance.sort_values(ascending=False).head(10)