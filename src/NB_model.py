
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Naive Bayes model search
def NB(x_train, y_train, x_validation, y_validation, x_test, y_test):
    NB_model = GaussianNB()
    NB_model.fit(x_train, y_train)
    NB_pred = NB_model.predict(x_test)
    NB_acc = accuracy_score(y_test, NB_pred)
    
    return NB_acc, NB_model
    