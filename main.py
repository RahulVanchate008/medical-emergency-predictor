from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.train_lstm_router import router as train_lstm_router
from routes.preprocessing_router import router as preprocessing_router

app = FastAPI()

app.include_router(preprocessing_router)
app.include_router(train_lstm_router)

# CORS settings
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

