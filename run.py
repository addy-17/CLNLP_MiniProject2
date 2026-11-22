import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting BPE-Trans: Low-Resource NMT Server...")
    print("🌐 Web Application: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
