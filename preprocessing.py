import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

def load_data(file_path):
    """
    Load dataset from file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, augment=False):
    """
    Preprocess the dataset.
    """
    # Özellikleri ve hedef değişkeni ayır
    X = data.drop("target", axis=1)
    y = data["target"]

    # Sınıf dağılımını kontrol et
    print("Class distribution before augmentation:")
    print(y.value_counts())

    # Sentetik veri oluşturma
    if augment:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    # Sınıf dağılımını kontrol et
    print("Class distribution after augmentation:")
    print(y.value_counts())

    # Eğitim ve test setlerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    # Standart ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def load_data_with_clusters(file_path):
    """
    Load dataset with clusters from file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data_with_clusters(data, augment=False):
    """
    Preprocess the dataset with clusters.
    """
    # Özellikleri ve hedef değişkeni ayır
    X = data.drop(["target", "Cluster"], axis=1)  # "Cluster" sütununu kullanacağız
    y = data["Cluster"]

    # Sınıf dağılımını kontrol et
    print("Class distribution before augmentation:")
    print(y.value_counts())

    # Sentetik veri oluşturma
    if augment:
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)

    # Sınıf dağılımını kontrol et
    print("Class distribution after augmentation:")
    print(y.value_counts())

    # Eğitim ve test setlerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    # Standart ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
