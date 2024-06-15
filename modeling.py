from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_model(X_train, y_train, multiclass=False):
    """
    Train CatBoost model.
    """
    if multiclass:
        model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='MultiClass')
    else:
        model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss')
    
    model.fit(X_train, y_train, verbose=100)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f'Model Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(cm)
    
    return accuracy, precision, recall, f1, cm
