import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.model import create_model, train_model
from src.metrics import calculate_metrics

def test_model_overfitting_check():
    """Тест проверки на переобучение модели"""
    # Загружаем данные
    X, y = load_iris(return_X_y=True)
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Создаем и обучаем модель
    model = create_model(random_state=42)
    trained_model = train_model(model, X_train, y_train)
    
    # Предсказания на тренировочных и тестовых данных
    y_pred_train = trained_model.predict(X_train)
    y_pred_test = trained_model.predict(X_test)
    
    # Вычисляем accuracy на train и test
    from sklearn.metrics import accuracy_score
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Difference: {abs(train_accuracy - test_accuracy):.4f}")
    
    # Проверяем что разница между train и test accuracy не слишком большая
    # Обычно приемлемой считается разница до 0.1 (10%)
    accuracy_difference = abs(train_accuracy - test_accuracy)
    assert accuracy_difference < 0.15, (
        f"Модель переобучается! "
        f"Разница между train и test accuracy: {accuracy_difference:.4f} > 0.15"
    )
    
    # Дополнительно: test accuracy должен быть достаточно высоким
    assert test_accuracy > 0.85, (
        f"Слишком низкое качество на тесте: {test_accuracy:.4f} < 0.85"
    )
