# iris_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar dados
data = load_iris()
X = data['data']
y = data['target']

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Avaliar o modelo
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Salvar o modelo
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)
