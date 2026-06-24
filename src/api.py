from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API de Previsão de Imóveis")

modelo = joblib.load("modelo/modelo.pkl")

class Imovel(BaseModel):
    area: float
    quartos: int
    banheiros: int
    garagem: int

@app.get("/")
def root():
    return {"status": "online", "modelo": "regressão linear — previsão de preço de imóveis"}

@app.post("/prever")
def prever(imovel: Imovel):
    dados = np.array([[imovel.area, imovel.quartos, imovel.banheiros, imovel.garagem]])
    preco = modelo.predict(dados)[0]
    return {
        "entrada": imovel.model_dump(),
        "preco_previsto": f"R$ {preco:,.0f}"
    }
