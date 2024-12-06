import pandas as pd

def load_data(file_path: str, sample_size: int = None, random_seed: int = 42):
    # Cargar el DataFrame
    df = pd.read_csv(file_path, sep=';')
    
    # Tomar una muestra si se especifica
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=random_seed)
    
    return df