# arcface_service.py
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2


class ArcFaceService:
    """
    Servicio para trabajar con el modelo ArcFace ONNX.

    - Lee el modelo y detecta automáticamente si la entrada es NCHW (1,3,112,112)
      o NHWC (1,112,112,3).
    - Preprocesa imágenes PIL (o rutas) a ese formato.
    - Detecta y recorta el rostro antes de pasar al modelo.
    - Entrega embeddings L2-normalizados.
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        model_path = str(model_path)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # típico: [1, 3, 112, 112] o [1,112,112,3]

        if len(self.input_shape) != 4:
            raise RuntimeError(
                f"Shape de entrada no soportada (esperado 4 dims): {self.input_shape}"
            )

        n, d1, d2, d3 = self.input_shape

        # Detectar si es NCHW (1, C, H, W) o NHWC (1, H, W, C)
        if d1 in (1, 3) and d2 == d3:
            # NCHW
            self.channels_first = True
            self.channels = int(d1)
            self.input_h = int(d2)
            self.input_w = int(d3)
        elif d3 in (1, 3) and d1 == d2:
            # NHWC
            self.channels_first = False
            self.channels = int(d3)
            self.input_h = int(d1)
            self.input_w = int(d2)
        else:
            raise RuntimeError(
                f"Shape de entrada no reconocida: {self.input_shape}"
            )

        if self.channels != 3:
            raise RuntimeError(
                f"El modelo espera {self.channels} canales, y asumimos RGB (3). Revisa el modelo."
            )

    # ---------- DETECCIÓN Y RECORTE DE ROSTRO ----------
    def detect_and_crop_face(self, img: Image.Image) -> Image.Image:
        """
        Detecta un rostro con HaarCascade y recorta solo la cara.

        Si no se detecta ningún rostro, lanza RuntimeError
        (para que la app muestre el mensaje y no deje marcar).
        """
        # Aseguramos RGB y convertimos a numpy
        rgb = img.convert("RGB")
        cv_img = np.array(rgb)              # RGB
        cv_img = cv_img[:, :, ::-1]         # RGB -> BGR (OpenCV)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        h, w = gray.shape[:2]

        # Si la imagen es muy grande, la bajamos para que la cascada funcione mejor
        max_side = max(h, w)
        scale = 1.0
        if max_side > 800:
            scale = 800.0 / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray_small = cv2.resize(gray, (new_w, new_h))
        else:
            gray_small = gray

        # Ruta del cascade: primero intentamos archivo local, si no, el de OpenCV
        local_cascade = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
        if local_cascade.exists():
            cascade_path = str(local_cascade)
        else:
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise RuntimeError(f"No se pudo cargar el cascade de caras en: {cascade_path}")

        # Parámetros algo tolerantes pero no tan flojos
        faces = face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(70, 70),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # Este mensaje se propaga hasta Flutter
            raise RuntimeError(
                "No se detectó ningún rostro en la imagen. "
                "Acércate, mira al frente y revisa la iluminación."
            )

        # Tomamos el rostro con mayor área
        x, y, w_box, h_box = max(faces, key=lambda b: b[2] * b[3])

        # Si reescalamos para detectar, volvemos a las coords originales
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w_box = int(w_box / scale)
            h_box = int(h_box / scale)

        # Recortamos el rostro sobre la imagen RGB de PIL
        face = rgb.crop((x, y, x + w_box, y + h_box))
        return face

    # ---------- Preproceso de imagen ----------
    def _preprocess(self, img: Image.Image) -> np.ndarray:
        # Aseguramos RGB
        img = img.convert("RGB")

        # Redimensionar al tamaño que el modelo espera
        img = img.resize((self.input_w, self.input_h), Image.BILINEAR)

        # H, W, 3 en [0,1]
        arr = np.asarray(img).astype("float32") / 255.0

        # Normalización típica de InsightFace / ArcFace
        mean = np.array([0.5, 0.5, 0.5], dtype="float32")
        std = np.array([0.5, 0.5, 0.5], dtype="float32")
        arr = (arr - mean) / std  # sigue H, W, 3

        # Para NCHW hay que pasar a [3, H, W]
        if self.channels_first:
            arr = np.transpose(arr, (2, 0, 1))  # 3, H, W

        # Añadimos batch: [1, C, H, W] o [1, H, W, C]
        arr = np.expand_dims(arr, axis=0)
        return arr

    # ---------- Embeddings ----------
    def get_embedding_from_path(self, img_path: Union[str, Path]) -> np.ndarray:
        img = Image.open(img_path)
        return self.get_embedding(img)

    def get_embedding(self, img_or_path: Union[Image.Image, str, Path]) -> np.ndarray:
        # Abre imagen si viene como ruta
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path)
        else:
            img = img_or_path

        # 1) Detectar y recortar el rostro
        img = self.detect_and_crop_face(img)

        # 2) Preprocesar + pasar por el modelo
        x = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: x})
        emb = outputs[0]

        # Quitamos batch: [D] o [1, D]
        emb = np.asarray(emb, dtype="float32")
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]

        # L2-normalización
        norm = np.linalg.norm(emb) + 1e-9
        emb = emb / norm
        return emb.astype("float32")

    # ---------- Utilidades ----------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype="float32").ravel()
        b = np.asarray(b, dtype="float32").ravel()
        a /= (np.linalg.norm(a) + 1e-9)
        b /= (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))

    def compare_embeddings(
        self, stored: np.ndarray, current: np.ndarray, agg: str = "max"
    ) -> Tuple[float, float]:
        """
        - stored: [N, D] o [D]
        - current: [D]
        Devuelve (sim_max, sim_mean)
        """
        stored = np.asarray(stored, dtype="float32")
        current = np.asarray(current, dtype="float32")

        if stored.ndim == 1:
            stored = stored[None, :]

        sims = []
        for row in stored:
            sims.append(self.cosine_similarity(row, current))
        sims = np.array(sims, dtype="float32")

        sim_max = float(sims.max())
        sim_mean = float(sims.mean())
        return (sim_max, sim_mean)
