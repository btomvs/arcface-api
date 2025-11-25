from pathlib import Path
from arcface_service import ArcFaceService

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "arcface.onnx"
IMG1_PATH = BASE_DIR / "face1.jpg"
IMG2_PATH = BASE_DIR / "face2.jpg"


def main():
    service = ArcFaceService(MODEL_PATH)

    result = service.compare_faces(IMG1_PATH, IMG2_PATH, threshold=0.6)

    print(f"Similitud: {result['similarity']:.4f}")
    print(f"Umbral: {result['threshold']}")
    print("¿Misma persona?:", "SÍ ✅" if result["is_same_person"] else "NO ❌")


if __name__ == "__main__":
    main()
