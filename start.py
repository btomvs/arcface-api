# start.py
import os
import uvicorn

def main():
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        # opcional: más workers si quieres
        # workers=1,
    )

if __name__ == "__main__":
    main()
