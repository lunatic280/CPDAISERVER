from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/detect/")
async def detect_image(request: Request):
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(content={"error": "이미지 디코딩 실패"}, status_code=400)

    results = model(img[..., ::-1], verbose=False)[0]

    detections = []
    person_detected = False

    for box, cls, conf in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.cls.cpu().numpy(),
        results.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        if label == "person":
            person_detected = True
        detections.append({
            "box": [x1, y1, x2, y2],
            "label": label,
            "confidence": float(conf)
        })

    # 사람이 감지되면 콘솔에 '사람' 출력
    if person_detected:
        print("사람")

    return {"detections": detections}