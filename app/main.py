from fastapi import FastAPI, File, UploadFile
from PIL import Image
from app.ai_service import predict_image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 미들웨어 설정
origins = [
    "http://localhost:3000",  # 로컬 개발 서버에서 요청을 허용할 경우
    "https://example.com",  # 특정 도메인에서 요청을 허용할 경우
    "*",  # 모든 도메인에서 요청을 허용할 경우
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 도메인
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 (GET, POST 등) 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        result = predict_image(image)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 서버 실행 명령: uvicorn app.main:app --reload
