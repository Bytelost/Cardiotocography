
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# KNN model search
def KNN_model_search(x_train, y_train, x_validation, y_validation):
    
    best_accuracy = -1
    
    # Test the different models
    for i in ("uniform", "distance"):
        for k in range(1, 50):
            knn_instance = KNeighborsClassifier(n_neighbors=k, weights=i)
            knn_instance.fit(x_train, y_train)
            pred = knn_instance.predict(x_validation)
            accuracy = accuracy_score(y_validation, pred)
            
            # Update the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_distance = i
                best_model = knn_instance
    
    return best_model, best_k, best_distance

# KNN model
def KNN(x_train, y_train, x_validation, y_validation, x_test, y_test):
    
    # Convert the data to contiguous arrays
    x_train = np.ascontiguousarray(x_train)
    y_train = np.ascontiguousarray(y_train)
    x_validation = np.ascontiguousarray(x_validation)
    y_validation = np.ascontiguousarray(y_validation)
    x_test = np.ascontiguousarray(x_test)
    y_test = np.ascontiguousarray(y_test)
    
    # Find the best model
    best_model, best_k, best_distance = KNN_model_search(x_train, y_train, x_validation, y_validation)
    
    # Test the best model
    KNN_pred = best_model.predict(x_test)
    KNN_acc = accuracy_score(y_test, KNN_pred)
    
    knn_params = {"n_neighbors": best_k, "weights": best_distance}
    
    return KNN_acc, best_model, knn_params