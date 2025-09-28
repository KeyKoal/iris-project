# src/metrics.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет метрики качества модели.
    Args:
        y_true: истинные значения
        y_pred: предсказанные значения

    Returns:
        accuracy: точность
        report: текстовый отчет с метриками
        cm: матрица ошибок
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, report, cm
