# start.py
import os
import uvicorn


def main():
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        # workers=1,  # opcional
    )


if __name__ == "__main__":
    main()
