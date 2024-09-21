import os
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import FileResponse
from predict import predict_from_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (bạn có thể thay "*" bằng một danh sách các domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các header
)

@app.get("/")
async def read_index():
    # Trả về file index.html
    path = os.path.join(os.getcwd(), "index.html")
    return FileResponse(path)
@app.get("/predict")
def get_prediction(tvadp: float, radadp: float, newsadp: float):
    # Gọi hàm predict_from_input với 3 tham số
    prediction = predict_from_input(tvadp, radadp, newsadp)
    return {"salesPrice": prediction}
