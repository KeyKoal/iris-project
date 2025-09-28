import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):  
    """Загружает данные из CSV файла"""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, target_column=None):  
    """Разделяет данные на признаки и целевую переменную"""
    if target_column is None:
        target_column = df.columns[-1]
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):  
    """Разделяет данные на тренировочную и тестовую выборки"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
