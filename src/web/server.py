"""
Web app backend (FastAPI): upload coin photo(s) -> prediction + catalogue.json info -> XAI heatmaps.

    GET  /                          frontend (src/web/static)
    GET  /api/methods               available XAI methods
    POST /api/predict               multipart: observe and/or reverse image -> prediction (with catalogue entry) + session_id
    POST /api/explain               json {session_id, side, method} -> raw image + heatmap overlay (data URLs)

Run with: python auto_commands/web_app.py
"""
import base64
import io
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from src.constants import EXPLAINERS, MODEL_CONFIGS
from src.engine.inference import CoinInference
from src.xai.core.visualization import overlay_heatmap

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_DISPLAY_SIZE = 768  # longest side of images sent back to the browser
MAX_SESSIONS = 50       # uploaded images kept in memory for later /api/explain calls

METHOD_INFO = {
    "gradcam": "Fast. Coarse map from the last conv block.",
    "gradcam++": "Fast. Grad-CAM variant, better with several important regions.",
    "lime": "Slow (~1000 forward passes). Superpixels that support / oppose the class.",
    "shap": "Slow. Pixel attributions vs. training images, needs the train split.",
}


class ExplainRequest(BaseModel):
    session_id: str
    side: str
    method: str


def _to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class CoinService:
    """Holds the models, uploaded images and lazily created explainers. One lock: the GPU is shared."""

    def __init__(self, observe_run, reverse_run, catalogue_path, data_root):
        self.inference = CoinInference(
            observe_run=observe_run,
            reverse_run=reverse_run,
            catalogue_path=catalogue_path,
            data_root=data_root,
        )
        self.data_root = data_root
        self.lock = threading.Lock()
        self.sessions = OrderedDict()  # session_id -> {"images": {side: PIL}, "class_idx": int}
        self.explainers = {}           # (side, method) -> explainer

    def predict(self, images):
        with self.lock:
            prediction = self.inference.predict(
                observe_image=images.get("Observe"), reverse_image=images.get("Reverse")
            )

        session_id = uuid.uuid4().hex
        self.sessions[session_id] = {
            "images": images,
            "class_idx": self.inference.class_numbers.index(prediction.class_number),
        }
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)
        return session_id, prediction

    def _explainer(self, side, method):
        key = (side, method)
        if key not in self.explainers:
            self.explainers[key] = EXPLAINERS[method](
                model=self.inference.models[side],
                device=self.inference.device,
                target_layer=MODEL_CONFIGS[self.inference.model_names[side]]["target_layer"],
                data_root=self.data_root,
                side=side,
            )
        return self.explainers[key]

    def explain(self, session_id, side, method):
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "Session expired, run the prediction again")
        if side not in session["images"]:
            raise HTTPException(400, f"No {side} image was uploaded in this session")
        if method not in EXPLAINERS:
            raise HTTPException(400, f"Unknown method '{method}', available: {list(EXPLAINERS)}")

        image = session["images"][side]
        started = time.perf_counter()
        with self.lock:
            try:
                explainer = self._explainer(side, method)
            except FileNotFoundError as e:
                raise HTTPException(400, f"{method} can't be initialized: {e}")
            tensor = self.inference._load_image(image)
            explanation = explainer.explain(tensor, session["class_idx"])

        display = image.copy()
        display.thumbnail((MAX_DISPLAY_SIZE, MAX_DISPLAY_SIZE))
        return {
            "side": side,
            "method": method,
            "method_name": explainer.name,
            "signed": explainer.signed,
            "class_number": self.inference.class_numbers[session["class_idx"]],
            "seconds": round(time.perf_counter() - started, 2),
            "raw": _to_data_url(display),
            "overlay": _to_data_url(overlay_heatmap(display, explanation, explainer.signed)),
        }


async def _read_image(upload):
    try:
        image = Image.open(io.BytesIO(await upload.read()))
        image = ImageOps.exif_transpose(image)  # phone photos: show and classify them upright
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"'{upload.filename}' is not a readable image")


def create_app(observe_run, reverse_run, catalogue_path="dataset/data/catalogue.json", data_root="dataset/data"):
    service = CoinService(observe_run, reverse_run, catalogue_path, data_root)
    app = FastAPI(title="Coin classification")

    @app.middleware("http")
    async def no_cache_frontend(request, call_next):
        # browsers must revalidate index.html / app.js / style.css, otherwise after an update
        # they keep running the old script against the new API
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/methods")
    def methods():
        return {
            "methods": [
                {"id": key, "name": cls.name, "signed": cls.signed, "info": METHOD_INFO.get(key, "")}
                for key, cls in EXPLAINERS.items()
            ],
            "sides": list(service.inference.models),
        }

    @app.post("/api/predict")
    async def predict(observe: UploadFile | None = File(None), reverse: UploadFile | None = File(None)):
        images = {}
        for side, upload in (("Observe", observe), ("Reverse", reverse)):
            if upload is not None and upload.filename:
                if side not in service.inference.models:
                    raise HTTPException(400, f"No {side} model is loaded")
                images[side] = await _read_image(upload)
        if not images:
            raise HTTPException(400, "Upload at least one image (Observe or Reverse)")

        session_id, prediction = await run_in_threadpool(service.predict, images)
        return {"session_id": session_id, "prediction": prediction.to_dict()}

    @app.post("/api/explain")
    def explain(request: ExplainRequest):
        return service.explain(request.session_id, request.side, request.method)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
