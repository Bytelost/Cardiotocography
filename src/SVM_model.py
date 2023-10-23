
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# SVM model search
def SVM_model_search(x_train, y_train, x_validation, y_validation):
    
    highest = -1
    
    # Test the different models
    for i in('linear', 'poly', 'rbf', 'sigmoid'):
        for j in (0.1, 1, 10):
            SVM = SVC(kernel=i, C=j)
            SVM.fit(x_train, y_train)
            pred = SVM.predict(x_validation)
            acc = accuracy_score(y_validation, pred)
            
            # Save the best model
            if acc > highest:
                best_model = SVM
                best_kernel = i
                best_c = j
                highest = acc
                
    return best_model, best_kernel, best_c


# SVM model
def SVM(x_train, y_train, x_validation, y_validation, x_test, y_test):
        
        # Find the best model
        best_model, best_kernel, best_c = SVM_model_search(x_train, y_train, x_validation, y_validation)
        
        # Test the best model
        SVM_pred = best_model.predict(x_test)
        SVM_acc = accuracy_score(y_test, SVM_pred)
        
        return SVM_acc, best_model, best_kernel, best_c