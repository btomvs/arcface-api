# face_store.py
from pathlib import Path
from typing import Optional

import numpy as np


class FaceStore:
  """
  Guarda embeddings por usuario en archivos .npy.

  - Cada usuario se guarda en embeddings/<user_id>.npy
  - El archivo contiene un array de shape [N, D] (varios ejemplos) o [D] (antiguo).
  - Al registrar de nuevo, se agrega un embedding y se limitan a max_samples.
  """

  def __init__(self, base_dir: Path, max_samples: int = 5) -> None:
    self.base_dir = Path(base_dir)
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self.max_samples = max_samples

  def _path_for(self, user_id: str) -> Path:
    # Evitar caracteres raros en el nombre de archivo
    safe_id = user_id.replace("/", "_").replace("\\", "_")
    return self.base_dir / f"{safe_id}.npy"

  # ---------- Guardar ----------
  def save_embedding(self, user_id: str, emb: np.ndarray) -> None:
    p = self._path_for(user_id)

    emb = np.asarray(emb, dtype="float32").ravel()[None, :]  # [1, D]

    if p.exists():
      prev = np.load(p)  # [N, D] o [D]
      if prev.ndim == 1:
        prev = prev[None, :]
      all_embs = np.vstack([prev, emb])  # [N+1, D]
      # Limitar a los últimos max_samples
      if all_embs.shape[0] > self.max_samples:
        all_embs = all_embs[-self.max_samples :]
    else:
      all_embs = emb  # [1, D]

    np.save(p, all_embs)

  # ---------- Leer todos los embeddings ----------
  def load_embeddings(self, user_id: str) -> Optional[np.ndarray]:
    p = self._path_for(user_id)
    if not p.exists():
      return None

    data = np.load(p)
    if data.ndim == 1:
      data = data[None, :]
    return data.astype("float32")

  # ---------- Borrar usuario (por si quieres re-enrolar “desde cero”) ----------
  def delete_embeddings(self, user_id: str) -> None:
    p = self._path_for(user_id)
    if p.exists():
      p.unlink()
