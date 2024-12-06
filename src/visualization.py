import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_visualizations(df, model, poly, scaler, y, X, X_test, y_test):
    # Crear figura con subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=3.0)

    # Predicciones con el modelo completo
    predictions = model.predict(poly.transform(scaler.transform(X)))
    
    # Predicciones para el conjunto de prueba
    y_test_pred = model.predict(poly.transform(X_test))

    # Dispersión: Edad vs. Predicción
    axs[0, 0].scatter(df['age'], y, alpha=0.5, label='Real')
    axs[0, 0].scatter(df['age'], predictions, alpha=0.5, label='Prediction', color='r')
    axs[0, 0].set_title('Age vs Cardiovascular Risk')
    axs[0, 0].set_xlabel('Age (days)')
    axs[0, 0].set_ylabel('Cardiovascular Risk')
    axs[0, 0].legend()

    # Dispersión: Colesterol vs. Predicción
    axs[0, 1].scatter(df['cholesterol'], y, alpha=0.5, label='Real')
    axs[0, 1].scatter(df['cholesterol'], predictions, alpha=0.5, label='Prediction', color='r')
    axs[0, 1].set_title('Cholesterol vs Cardiovascular Risk')
    axs[0, 1].set_xlabel('Cholesterol Level')
    axs[0, 1].set_ylabel('Cardiovascular Risk')
    axs[0, 1].legend()

    # Gráfico de errores: Predicción vs Real (muestra de prueba)
    axs[1, 0].plot(y_test.values[:100], label='Actual', color='blue', linestyle='dotted')
    axs[1, 0].plot(y_test_pred[:100], label='Predicted', color='orange', linestyle='solid')
    axs[1, 0].set_title('Actual vs Predicted (Sample)')
    axs[1, 0].set_xlabel('Sample Index')
    axs[1, 0].set_ylabel('Cardiovascular Risk')
    axs[1, 0].legend()

    # Gráfico de importancia de factores
    importance = np.abs(model.coef_[:10])
    features = poly.get_feature_names_out(input_features=X.columns)[:10]
    axs[1, 1].barh(features, importance, color='green')
    axs[1, 1].set_title('Feature Importance')
    axs[1, 1].set_xlabel('Importance')

    plt.tight_layout()
    plt.show()