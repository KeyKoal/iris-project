from sklearn.ensemble import RandomForestClassifier


def create_model(n_estimators=100, max_depth=3, random_state=42):
    """Создает модель Random Forest с заданными параметрами"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    return model


def train_model(model, X_train, y_train):
    """Обучает модель на тренировочных данных"""
    trained_model = model.fit(X_train, y_train)
    return trained_model


def predict_model(model, X_test):
    """Делает предсказания на тестовых данных"""
    predictions = model.predict(X_test)
    return predictions
