# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:03:21 2024

@author: emirh
"""



from preprocessing import load_data, preprocess_data, load_data_with_clusters, preprocess_data_with_clusters
from modeling import train_model, evaluate_model

def main():
    # Normal veri setini yükleme ve işleme
    file_path = "C:/Users/emirh/Desktop/heart.csv"
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data, augment=True)  # Sentetik veri ekleme

    # Modeli eğitme ve değerlendirme
    model_normal = train_model(X_train, y_train)
    print("Results with Normal Data:")
    accuracy_normal, precision_normal, recall_normal, f1_normal, cm_normal = evaluate_model(model_normal, X_test, y_test)

    # Kümeleme veri setini yükleme ve işleme
    cluster_file_path = "C:/Users/emirh/Desktop/heart_with_clusters.csv"
    cluster_data = load_data_with_clusters(cluster_file_path)
    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = preprocess_data_with_clusters(cluster_data, augment=True)  # Sentetik veri ekleme

    # Modeli eğitme ve değerlendirme (multi-class)
    model_cluster = train_model(X_train_cluster, y_train_cluster, multiclass=True)
    print("Results with Clustered Data:")
    accuracy_clustered, precision_clustered, recall_clustered, f1_clustered, cm_clustered = evaluate_model(model_cluster, X_test_cluster, y_test_cluster)

    # Sonuçları karşılaştırma
    print("Summary of Results:")
    print(f"Accuracy with Normal Data: {accuracy_normal}")
    print(f"Precision with Normal Data: {precision_normal}")
    print(f"Recall with Normal Data: {recall_normal}")
    print(f"F1 Score with Normal Data: {f1_normal}")
    print(f"Confusion Matrix with Normal Data:\n{cm_normal}")

    print(f"Accuracy with Clustered Data: {accuracy_clustered}")
    print(f"Precision with Clustered Data: {precision_clustered}")
    print(f"Recall with Clustered Data: {recall_clustered}")
    print(f"F1 Score with Clustered Data: {f1_clustered}")
    print(f"Confusion Matrix with Clustered Data:\n{cm_clustered}")

if __name__ == "__main__":
    main()
