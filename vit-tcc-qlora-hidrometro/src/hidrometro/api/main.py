"""API FastAPI do pipeline híbrido."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from hidrometro.pipeline.hybrid import HybridPipeline

_pipeline: HybridPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = HybridPipeline()
    _pipeline.vlm.load()

    yield
    _pipeline = None


def create_app() -> FastAPI:
    app = FastAPI(title="Hidrômetro VLM API", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )    

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if _pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline não inicializado.")

        content = await file.read()
        image_array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Imagem inválida.")

        try:
            result = _pipeline.run(image)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return _pipeline.to_response_dict(result)

    return app


app = create_app()
