# tests/test_data_processing.py
import pandas as pd
import numpy as np
import pytest
from src.data_processing import load_data, preprocess_data, split_data

def test_no_missing_values_in_dataset():
    """Тест что в датасете нет пропущенных значений"""
    # Загружаем реальный датасет
    df = load_data('data/iris.csv')
    
    # Проверяем отсутствие пропусков во всем DataFrame
    missing_values = df.isnull().sum().sum()
    assert missing_values == 0, f"Найдено {missing_values} пропущенных значений в датасете"
    
    # Дополнительно: проверяем отсутствие пропусков в каждом столбце
    for column in df.columns:
        missing_in_column = df[column].isnull().sum()
        assert missing_in_column == 0, f"В столбце '{column}' найдено {missing_in_column} пропущенных значений"


