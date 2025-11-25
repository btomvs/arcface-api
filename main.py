# main.py
from pathlib import Path
import shutil
import tempfile
import math

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from arcface_service import ArcFaceService
from face_store import FaceStore

app = FastAPI(title="ArcFace TuTurno API")

BASE_DIR = Path(__file__).resolve().parent

# Modelo ArcFace ONNX (garavv/arcface-onnx)
MODEL_PATH = BASE_DIR / "models" / "arcface.onnx"

# Carpeta donde se guardan los embeddings .npy
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

# Servicios globales
arcface_service = ArcFaceService(model_path=MODEL_PATH)
face_store = FaceStore(base_dir=EMBEDDINGS_DIR, max_samples=5)


def _save_temp_file(upload: UploadFile) -> Path:
    """Guarda UploadFile en disco temporalmente y devuelve Path."""
    suffix = Path(upload.filename or "image.jpg").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp, upload.file as f:
        shutil.copyfileobj(f, tmp)
    return Path(tmp.name)


# ==============================
#      ENDPOINT: register-face
# ==============================
@app.post("/register-face")
async def register_face(
    user_id: str = Form(...),
    image: UploadFile = File(...),
):
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="La imagen debe ser JPG o PNG.",
        )

    tmp_path = _save_temp_file(image)

    try:
        # Embedding de esta captura
        emb = arcface_service.get_embedding(tmp_path)

        # Guardar / actualizar las muestras de este usuario
        face_store.save_embedding(user_id, emb)

        print(f"[REGISTER] user_id={user_id} -> embedding guardado.")
        return {"ok": True, "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {e}",
        )
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


# ==============================
#      ENDPOINT: verify-face
#  (SÓLO similitud, sin liveness)
# ==============================
@app.post("/verify-face")
async def verify_face(
    user_id: str = Form(...),
    image: UploadFile = File(...),
    # dejamos el parámetro para compatibilidad, pero usamos nuestros umbrales internos
    threshold: float = Form(0.90),
):
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="La imagen debe ser JPG o PNG.",
        )

    stored_embs = face_store.load_embeddings(user_id)
    if stored_embs is None:
        raise HTTPException(
            status_code=404,
            detail="No hay embedding registrado para este user_id.",
        )

    tmp_path = _save_temp_file(image)

    # UMBRALES ROBUSTOS (ajústalos aquí si hace falta)
    HARD_MAX = 0.95   # mínimo para la mejor coincidencia
    HARD_MEAN = 0.92  # mínimo para el promedio de coincidencias

    try:
        current_emb = arcface_service.get_embedding(tmp_path)

        # compare_embeddings devuelve (sim_max, sim_mean)
        sim_max, sim_mean = arcface_service.compare_embeddings(
            stored_embs, current_emb
        )

        samples_count = (
            int(stored_embs.shape[0]) if stored_embs.ndim > 1 else 1
        )

        print(
            f"[VERIFY] user_id={user_id} "
            f"samples={samples_count} "
            f"sim_max={sim_max:.4f} sim_mean={sim_mean:.4f} "
            f"thr_max={HARD_MAX} thr_mean={HARD_MEAN}"
        )

        # Decisión estricta: se deben cumplir *ambas* condiciones
        if not ((sim_max >= HARD_MAX) and (sim_mean >= HARD_MEAN)):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "FACE_NOT_MATCH",
                    "user_id": user_id,
                    "similarity_max": sim_max,
                    "similarity_mean": sim_mean,
                    "threshold_max": HARD_MAX,
                    "threshold_mean": HARD_MEAN,
                    "samples_count": samples_count,
                },
            )

        # Si llegó aquí, la cara se considera válida
        return {
            "user_id": user_id,
            "similarity_max": sim_max,
            "similarity_mean": sim_mean,
            "threshold_max": HARD_MAX,
            "threshold_mean": HARD_MEAN,
            "samples_count": samples_count,
            "is_same_person": True,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {e}",
        )
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


# =============================================
#  NUEVO ENDPOINT: verify-face-liveness
#  - Recibe 2 fotos (frente y con movimiento)
#  - Verifica identidad + movimiento del rostro
# =============================================
@app.post("/verify-face-liveness")
async def verify_face_liveness(
    user_id: str = Form(...),
    image_front: UploadFile = File(...),
    image_move: UploadFile = File(...),
):
    """
    Verificación con prueba de vida simple:
      - Debe existir un rostro en ambas imágenes.
      - El rostro debe haberse movido una cantidad mínima.
      - Ambas imágenes deben coincidir con el embedding almacenado.
    """
    for img in (image_front, image_move):
        if img.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            raise HTTPException(
                status_code=400,
                detail="Las imágenes deben ser JPG o PNG.",
            )

    stored_embs = face_store.load_embeddings(user_id)
    if stored_embs is None:
        raise HTTPException(
            status_code=404,
            detail="No hay embedding registrado para este user_id.",
        )

    tmp_front = _save_temp_file(image_front)
    tmp_move = _save_temp_file(image_move)

    # Umbrales para identidad (más altos que el endpoint simple)
    HARD_MAX = 0.985
    HARD_MEAN = 0.975

    # Umbral para movimiento relativo (15% del tamaño de la cara)
    MIN_MOVE_RATIO = 0.15

    try:
        # ---- 1) Liveness: detectar rostro y medir movimiento
        cx1, cy1, w1, h1 = arcface_service.detect_face_box_from_path(tmp_front)
        cx2, cy2, w2, h2 = arcface_service.detect_face_box_from_path(tmp_move)

        move_px = math.hypot(cx2 - cx1, cy2 - cy1)
        face_ref = (w1 + h1 + w2 + h2) / 4.0
        move_ratio = move_px / max(face_ref, 1e-6)

        liveness_ok = move_ratio >= MIN_MOVE_RATIO

        # ---- 2) Identidad: ambas fotos deben coincidir con el usuario
        emb_front = arcface_service.get_embedding(tmp_front)
        emb_move = arcface_service.get_embedding(tmp_move)

        sim1_max, sim1_mean = arcface_service.compare_embeddings(stored_embs, emb_front)
        sim2_max, sim2_mean = arcface_service.compare_embeddings(stored_embs, emb_move)

        # nos quedamos con el peor caso (para ser más estrictos)
        sim_max = min(sim1_max, sim2_max)
        sim_mean = min(sim1_mean, sim2_mean)

        samples_count = (
            int(stored_embs.shape[0]) if stored_embs.ndim > 1 else 1
        )

        print(
            f"[VERIFY_LIVENESS] user_id={user_id} "
            f"samples={samples_count} "
            f"sim1_max={sim1_max:.4f} sim2_max={sim2_max:.4f} "
            f"sim_max={sim_max:.4f} sim_mean={sim_mean:.4f} "
            f"move_ratio={move_ratio:.3f}"
        )

        identity_ok = (sim_max >= HARD_MAX) and (sim_mean >= HARD_MEAN)

        if not liveness_ok:
            # No hay movimiento suficiente del rostro
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "LIVENESS_FAIL",
                    "user_id": user_id,
                    "move_ratio": move_ratio,
                    "min_move_ratio": MIN_MOVE_RATIO,
                    "similarity_max": sim_max,
                    "similarity_mean": sim_mean,
                    "threshold_max": HARD_MAX,
                    "threshold_mean": HARD_MEAN,
                    "samples_count": samples_count,
                },
            )

        if not identity_ok:
            # Movimiento ok, pero la cara no coincide con el usuario
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "FACE_NOT_MATCH",
                    "user_id": user_id,
                    "move_ratio": move_ratio,
                    "min_move_ratio": MIN_MOVE_RATIO,
                    "similarity_max": sim_max,
                    "similarity_mean": sim_mean,
                    "threshold_max": HARD_MAX,
                    "threshold_mean": HARD_MEAN,
                    "samples_count": samples_count,
                },
            )

        # Si llegó aquí, pasó liveness + identidad
        return {
            "user_id": user_id,
            "is_same_person": True,
            "liveness_ok": True,
            "similarity_max": sim_max,
            "similarity_mean": sim_mean,
            "threshold_max": HARD_MAX,
            "threshold_mean": HARD_MEAN,
            "samples_count": samples_count,
            "move_ratio": move_ratio,
            "min_move_ratio": MIN_MOVE_RATIO,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {e}",
        )
    finally:
        for p in (tmp_front, tmp_move):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# Opcional: healthcheck
@app.get("/")
async def root():
    return {"status": "ok"}
