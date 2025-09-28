import pandas as pd
from src.data_processing import preprocess_data, split_data


def test_preprocess_data_removes_target_column():
    """Тест что целевая переменная удаляется из признаков"""
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'species': ['setosa', 'versicolor', 'virginica']
    })

    X, y = preprocess_data(test_data, 'species')
    assert 'species' not in X.columns


def test_split_data_proportions():
    """Тест корректности разделения данных"""
    X = pd.DataFrame({'feature': range(100)})
    y = pd.Series(range(100))

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_train) == 80
    assert len(X_test) == 20
