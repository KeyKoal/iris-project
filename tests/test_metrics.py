# tests/test_metrics.py
import numpy as np
from src.metrics import calculate_metrics

def test_confusion_matrix_values():
    """Тест правильности вычисления значений в матрице ошибок"""
    # Искусственные данные где мы знаем правильные значения матрицы
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])  # 4 нуля, 3 единицы, 3 двойки
    y_pred = np.array([0, 0, 1, 2, 1, 1, 2, 2, 2, 0])  # Предсказания с ошибками
    
        
    accuracy, report, cm = calculate_metrics(y_true, y_pred)
    
    # Проверяем конкретные значения в матрице ошибок
    assert cm[0, 0] == 2  # True class 0, Pred class 0
    assert cm[0, 1] == 1  # True class 0, Pred class 1
    assert cm[0, 2] == 1  # True class 0, Pred class 2
    
    assert cm[1, 0] == 0  # True class 1, Pred class 0
    assert cm[1, 1] == 2  # True class 1, Pred class 1
    assert cm[1, 2] == 1  # True class 1, Pred class 2
    
    assert cm[2, 0] == 1  # True class 2, Pred class 0
    assert cm[2, 1] == 0  # True class 2, Pred class 1
    assert cm[2, 2] == 2  # True class 2, Pred class 2
    
    # Проверяем что сумма всех элементов = количеству samples
    assert cm.sum() == len(y_true) == 10
    
    # Проверяем accuracy вручную: (2 + 2 + 2) / 10 = 0.6
    expected_accuracy = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / cm.sum()
    assert abs(accuracy - expected_accuracy) < 1e-10
    
    print("Матрица ошибок:")
    print(cm)
    print(f"Accuracy: {accuracy:.2f}")
