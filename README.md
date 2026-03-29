# Smart Crop Doctor 🌾

AI-Powered Plant Health Diagnostics

Smart Crop Doctor is an end-to-end intelligent agricultural application that detects and classifies crop diseases from leaf images or a real-time webcam feed. The system uses a Deep Learning model (YOLOv8) to provide accurate identification, assess the severity of the infection, and offer actionable treatment recommendations in English, Telugu, and Kannada.

## Features

- **🔍 Image Upload & Analysis:** Upload an image of a leaf to get instant analysis regarding disease, confidence, and severity.
- **📷 Real-Time Stream:** Point your webcam at a leaf to get live disease detection and bounding boxes directly in the browser.
- **🌐 Multi-Language Support:** Instant translation of disease descriptions and treatment plans into:
  - English
  - తెలుగు (Telugu)
  - ಕನ್ನಡ (Kannada)
- **📊 History & Reporting:** All predictions are logged. You can view your recent scans on the dashboard or download a CSV report.

## Tech Stack

- **Backend / API:** FastAPI, Python
- **Machine Learning:** Ultralytics YOLOv8, OpenCV
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Templates:** Jinja2

## Setup & Installation

Follow these steps to run the Smart Crop Doctor locally:

### 1. Requirements

Ensure you have Python 3.9+ installed.

### 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

*(Note: dependencies include `fastapi`, `uvicorn`, `python-multipart`, `ultralytics`, `opencv-python-headless`, and `jinja2`)*

### 3. Model Weights

The system expects a YOLOv8 model file. It will look for `models/best.pt` by default. If it is not found, it will automatically fall back to downloading and using `yolov8n.pt` for demo purposes.

## Running the Application

Start the FastAPI backend server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

Then open your browser and navigate to:

```
http://127.0.0.1:8000
```

## How to Use

1. **Upload an Image**
   Navigate to the "Upload Image" tab. Click or drag-and-drop an image of an infected leaf. The system will process it and show the detected condition, confidence, severity, and treatment plan. Use the language selector in the top right to change the language of the description and recommendations.
2. **Use the Live Camera**
   Navigate to the "Live Camera" tab and click "Start Camera". Present a leaf to your webcam, and the AI will try to detect any diseases in real-time.
3. **View History**
   Navigate to the "History" tab to see previous inferences. You can also download a CSV report of this data.

---

### Disclaimer

*The recommendations provided by this tool base their outcome on computer vision models and are meant for guidance. For critical agricultural decisions, please consult a local agricultural expert.*
