import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = load_iris()
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

model = RandomForestClassifier()
model.fit(x_train, y_train)

joblib.dump(model, 'modelo_treinado.pkl')