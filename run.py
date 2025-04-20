import uvicorn
import os
import sys

# src klasörünü Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8010,
        reload=True
    ) 