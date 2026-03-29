import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.detector import CropDetector
from app.utils import log_prediction, get_history, HISTORY_FILE

app = FastAPI(title="Smart Crop Doctor API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Detector
# This will load best.pt if available, else yolov8n.pt for demo
detector = CropDetector()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Renders the main dashboard UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Receives uploaded image, runs inference, and returns detection results."""
    contents = await file.read()
    
    # Predict
    img_base64, results = detector.predict_image(contents)
    
    if img_base64 is None:
        return {"error": results.get("error", "Failed to process image")}
        
    # Extracted translated info
    disease_info = results['disease_info']
    
    # Log prediction history (using English name for history logs)
    log_prediction(
        disease=disease_info['name']['en'],
        confidence=results['confidence'],
        severity=results['severity']
    )
    
    return {
        "image": img_base64,
        "disease_key": results['disease_key'],
        "disease_info": disease_info,
        "confidence": results['confidence'],
        "severity": results['severity']
    }

def generate_video_frames():
    """Generator for reading webcam and streaming MJPEG with inference."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
        
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            # Process frame with YOLO
            processed_frame = detector.process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()

@app.get("/video_feed")
def video_feed():
    """Endpoint for webcam live stream."""
    return StreamingResponse(generate_video_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/history")
def history_api():
    """Returns the detection history JSON."""
    return {"history": get_history()}

@app.get("/download_report")
def download_report():
    """Endpoint to download the CSV history."""
    import os
    if os.path.exists(HISTORY_FILE):
        return FileResponse(HISTORY_FILE, media_type="text/csv", filename="Crop_Disease_Report.csv")
    return {"error": "Report not found. Generate some predictions first."}
