import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("dados/imoveis.csv")

X = df[["area", "quartos", "banheiros", "garagem"]]
y = df["preco"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste)

print(f"R²: {r2_score(y_teste, y_pred):.3f}")
print(f"Erro médio absoluto: R$ {mean_absolute_error(y_teste, y_pred):,.0f}")

joblib.dump(modelo, "modelo/modelo.pkl")
print("Modelo salvo em modelo/modelo.pkl")
