from src.data_loader import load_data
from src.preprocessing import prepare_features, split_data, transform_features
from src.model import train_model, evaluate_model, optimize_model, identify_risk_factors
from src.visualization import create_visualizations

def main():
    # 1. Cargar datos
    file_path = 'data/cardio_train.csv'
    df = load_data(file_path, sample_size=1000)

    # 2. Preparar características
    feature_columns = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    X_scaled, y, scaler = prepare_features(df, feature_columns)

    # 3. Dividir datos
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 4. Transformar características
    X_train_poly, X_test_poly, poly = transform_features(X_train, X_test)

    # 5. Entrenar modelo
    model = train_model(X_train_poly, y_train)

    # 6. Evaluar modelo
    metrics = evaluate_model(model, X_train_poly, X_test_poly, y_train, y_test)
    print("Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 7. Optimizar modelo
    model = optimize_model(model, X_train_poly, y_train)

    # 8. Identificar factores de riesgo
    top_factors = identify_risk_factors(model, poly, df[feature_columns])
    print("\nTop 10 Risk Factors:")
    print(top_factors)

    # 9. Visualizar resultados
    create_visualizations(
        df, 
        model, 
        poly, 
        scaler, 
        y, 
        df[feature_columns],
        X_test,  # Añadido
        y_test   # Añadido
    )

if __name__ == "__main__":
    main()