from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import predict_text

app = FastAPI(title="Hierarchical Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Input(BaseModel):
    nama_produk: str
    deskripsi: str = ""

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: Input):
    result = predict_text(data.nama_produk, data.deskripsi)
    return {
        "success": True,
        "data": {
            "category_id": result
        }
    }