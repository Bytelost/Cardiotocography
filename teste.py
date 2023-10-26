# Call the libraries
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as ptl

# Load the database
data = pd.read_csv("CTG.csv")

# Delete the first column
data = data.drop(data.columns[0], axis=1)

df_data = pd.DataFrame(data)
#df_data = sk.utils.shuffle(df_data)
df_data = df_data.drop(columns=["NSP"])


from sklearn.model_selection import train_test_split

# Separate 50% of the data for training
x_train, x_temp, y_train, y_temp = train_test_split(df_data, data["NSP"], test_size=0.5, random_state=42)

# Separate 25% of the data for validation and 25% for testing
x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Train the model
NB = GaussianNB()
NB.fit(x_train, y_train)
pred = NB.predict(x_validation)
print("Acurácia:", accuracy_score(y_validation, pred))

pred = NB.predict(x_test)
acc = accuracy_score(y_test, pred)
print("Acurácia Teste:", acc)
confusion_matrix(y_test, pred)

