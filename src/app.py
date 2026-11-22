import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from src.data_loader import get_available_languages, LANG_MAP
from src.inference import TranslationEngine
from src.train import train_model

app = FastAPI(title="BPE-Trans NMT", description="Low-Resource Neural Machine Translation")

# Templates and static files
templates = Jinja2Templates(directory="templates")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared translation engine
engine = TranslationEngine()

# Training status tracking
training_jobs = {}

class TranslationRequest(BaseModel):
    text: str
    lang_key: str = "karbi"

class TrainRequest(BaseModel):
    lang_key: str
    epochs: Optional[int] = 12
    max_samples: Optional[int] = 1200

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "languages": get_available_languages()
    })

@app.get("/api/languages")
async def list_languages():
    return {"languages": get_available_languages()}

@app.get("/api/models")
async def list_models():
    models = engine.list_available_models()
    return {"models": models}

@app.post("/api/translate")
async def translate_text(req: TranslationRequest):
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "Text cannot be empty."})
    
    try:
        result = engine.translate(req.text.strip(), lang_key=req.lang_key)
        return result
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model for '{req.lang_key}' is not trained yet. Use the training panel to train it first."}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def run_background_train(lang_key: str, epochs: int, max_samples: int):
    training_jobs[lang_key] = {"status": "training"}
    try:
        cfg = train_model(
            lang_key=lang_key,
            max_samples=max_samples,
            vocab_size=600,
            epochs=epochs,
            batch_size=32,
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        training_jobs[lang_key] = {"status": "completed", "config": cfg}
        # Invalidate engine cache so new weights load
        if lang_key in engine.loaded_models:
            del engine.loaded_models[lang_key]
    except Exception as e:
        training_jobs[lang_key] = {"status": "failed", "error": str(e)}

@app.post("/api/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if req.lang_key not in LANG_MAP:
        return JSONResponse(status_code=400, content={"error": f"Invalid language key: {req.lang_key}"})

    if training_jobs.get(req.lang_key, {}).get("status") == "training":
        return {"message": f"Model training for '{req.lang_key}' is already in progress!"}

    background_tasks.add_task(run_background_train, req.lang_key, req.epochs or 10, req.max_samples or 1200)
    return {"message": f"Training started for {req.lang_key.upper()} in the background!"}

@app.get("/api/train/status/{lang_key}")
async def get_train_status(lang_key: str):
    return training_jobs.get(lang_key, {"status": "not_started"})

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
