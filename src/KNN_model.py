
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN model search
def KNN_model_search(x_train, y_train, x_validation, y_validation):
    
    # Define the hyperparameters to search
    best_accuracy = -1
    best_k = -1
    best_distance = -1
    best_model = None
    
    # Find the best hyperparameters
    for i in ("uniform", "distance"):
        for k in range(1, 50):
            knn_instance = KNeighborsClassifier(n_neighbors=k, weights=i)
            knn_instance.fit(x_train, y_train)
            pred = knn_instance.predict(x_validation)
            accuracy = accuracy_score(y_validation, pred)
            
            # Update the best hyperparameters
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_distance = i
                best_model = knn_instance
    
    return best_model, best_k, best_distance

# KNN model
def KNN(x_train, y_train, x_validation, y_validation, x_test, y_test):
    
    # Find the best model and hyperparameters
    best_model, best_k, best_distance = KNN_model_search(x_train, y_train, x_validation, y_validation)
    
    # Test the best model
    KNN_pred = KNN.predict(x_test)
    KNN_acc = accuracy_score(y_test, KNN_pred)
    
    return KNN_acc, best_model, best_k, best_distance