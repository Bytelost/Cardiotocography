
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# MLP model search
def MLP_model_search(x_train, y_train, x_validation, y_validation):
    
    higher_acc = -1
    hiden_layer = [(100, 50, 25), (21, 14, 1), (100, 100, 100)]

    # Find the best activation function
    for i in ('logistic', 'tanh', 'relu'):
            
        # Find the best learning rate
        for k in ('constant', 'invscaling', 'adaptive'):
            
            # Find the best number of iteractions
            for l in range(1000, 2000, 100):
                
                # Find the best number of hidden layers
                for h in hiden_layer:
                    
                    # Create the model
                    clf = MLPClassifier(hidden_layer_sizes=h, activation=i, solver='adam', learning_rate=k, max_iter=l)
                    clf.fit(x_train, y_train)
                    pred = clf.predict(x_validation)
                    acc = accuracy_score(y_validation, pred)
                    
                    # Save the best model
                    if acc > higher_acc:
                        best_acc = acc
                        best_mlp = clf
                        best_activation = i
                        best_learning_rate = k
                        best_iteractions = l
                        best_hidden_layer = h
                        
    return best_acc, best_mlp, best_activation, best_learning_rate, best_iteractions, best_hidden_layer

# MLP model
def MLP(x_train, y_train, x_validation, y_validation, x_test, y_test):
        
        # Find the best model
        best_acc, best_mlp, best_activation, best_learning_rate, best_iteractions, best_hidden_layer = MLP_model_search(x_train, y_train, x_validation, y_validation)
        
        # Test the best model
        MLP_pred = best_mlp.predict(x_test)
        MLP_acc = accuracy_score(y_test, MLP_pred)
        
        best_params = {"activation": best_activation, "learning_rate": best_learning_rate, "iteractions": best_iteractions, "hidden_layer": best_hidden_layer}
        
        return best_acc, best_mlp, best_params