
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Decision Tree model search
def DT_model_search(x_train, y_train, x_validation, y_validation):
    
    higher_acc = -1
    
    # Test the different models
    for i in ("gini", "entropy"):
        for j in ("best", "random"):
            for k in range(5, 21):
                for l in range(2, 20):
                    DT = DecisionTreeClassifier(criterion=i, splitter=j, max_depth=k, min_samples_leaf=l)
                    DT.fit(x_train, y_train)
                    pred = DT.predict(x_validation)
                    
                    # Save the best model
                    if accuracy_score(y_validation, pred) > higher_acc:
                        best_model = DT
                        best_criterion = i
                        best_splitter = j
                        best_depth = k
                        best_leaf = l
                        higher_acc = accuracy_score(y_validation, pred)
        
        return best_model, best_criterion, best_splitter, best_depth, best_leaf
    
    
# Decision Tree model
def DT(x_train, y_train, x_validation, y_validation, x_test, y_test):
    
    # Finda the best model
    best_model, best_criterion, best_splitter, best_depth, best_leaf = DT_model_search(x_train, y_train, x_validation, y_validation)
    
    # Test the best model
    DT_pred = best_model.predict(x_test)
    DT_acc = accuracy_score(y_test, DT_pred)
    
    best_params = {"criterion": best_criterion, "splitter": best_splitter, "max_depth": best_depth, "min_samples_leaf": best_leaf}
    
    return DT_acc, best_model, best_params