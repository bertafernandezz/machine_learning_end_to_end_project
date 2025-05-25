from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import pickle
from utils.transformations import ExtendedTransformation, SimpleTransformation
from utils.filters import SimpleFilter
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Inicialización de FastAPI
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------------
# Definición de entrada esperada
# -------------------------------
class PredictRequest(BaseModel):
    X: dict

# -------------------------------
# Variables globales
# -------------------------------
preproccesor = None
filter = None
model_80 = None
model_90 = None
model_99 = None

# -------------------------------
# Endpoint para cargar modelos
# -------------------------------
@app.post("/cargar_modelos")
async def cargar_modelos(
    file_pre: UploadFile = File(...),
    file_filter: UploadFile = File(...),
    file_model_80: UploadFile = File(...),
    file_model_90: UploadFile = File(...),
    file_model_99: UploadFile = File(...),
):
    global preproccesor, filter, model_80, model_90, model_99

    files = [file_pre, file_filter, file_model_80, file_model_90, file_model_99]
    if any([not f.filename.endswith(".pkl") for f in files]):
        raise HTTPException(status_code=400, detail="Todos los archivos deben ser .pkl")

    try:
        preproccesor = pickle.loads(await file_pre.read())
        filter = pickle.loads(await file_filter.read())
        model_80 = pickle.loads(await file_model_80.read())
        model_90 = pickle.loads(await file_model_90.read())
        model_99 = pickle.loads(await file_model_99.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar modelos: {e}")

    return {"status": "Modelos cargados correctamente."}

# -------------------------------
# Función genérica para predicción
# -------------------------------
def hacer_prediccion(model, request):
    if any([model is None, preproccesor is None, filter is None]):
        raise HTTPException(status_code=503, detail="Modelos no cargados.")

    try:
        x_dict = request.X
        x_pd = pd.DataFrame(x_dict)
        x_transform = preproccesor.transform(x_pd)
        x_filtered, _ = filter.transform(x_transform, None)

        y_pred, intervals = model.predict(x_filtered)
        y_pred_un = preproccesor.inverse_transform(y_pred.reshape(-1, 1)).tolist()
        y_low = preproccesor.inverse_transform(intervals[:, 0]).tolist()
        y_high = preproccesor.inverse_transform(intervals[:, 1]).tolist()

        return {
            "y_pred": y_pred_un,
            "y_low": y_low,
            "y_high": y_high,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Endpoints para predicción
# -------------------------------
@app.post("/predict_80")
async def predict_80(request: PredictRequest):
    return hacer_prediccion(model_80, request)

@app.post("/predict_90")
async def predict_90(request: PredictRequest):
    return hacer_prediccion(model_90, request)

@app.post("/predict_99")
async def predict_99(request: PredictRequest):
    return hacer_prediccion(model_99, request)
