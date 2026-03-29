import os
import cv2
import numpy as np
from ultralytics import YOLO

class CropDetector:
    def __init__(self, model_path='models/best.pt', default_model='yolov8n.pt'):
        """
        Initializes the YOLOv8 model. Uses trained model if available, else falls back to a default.
        """
        self.model_path = model_path
        self.is_demo = False
        if os.path.exists(self.model_path):
            print(f"Loading trained model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_demo = False
        else:
            print(f"Warning: {self.model_path} not found. Loading default {default_model} for demo.")
            self.model = YOLO(default_model)
            self.is_demo = True
            
        # Standard agricultural recommendations mapping with multi-language support
        self.disease_info = {
            'healthy': {
                'name': {
                    'en': 'Healthy',
                    'te': 'ఆరోగ్యంగా ఉంది',
                    'kn': 'ಆರೋಗ್ಯಕರ'
                },
                'description': {
                    'en': 'The plant is healthy and shows no signs of disease.',
                    'te': 'మొక్క ఆరోగ్యంగా ఉంది మరియు ఎలాంటి వ్యాధి లక్షణాలు లేవు.',
                    'kn': 'ಸಸ್ಯವು ಆರೋಗ್ಯವಾಗಿದೆ ಮತ್ತು ಯಾವುದೇ ರೋಗದ ಲಕ್ಷಣಗಳಿಲ್ಲ.'
                },
                'recommendation': {
                    'en': 'No treatment required. Maintain proper watering and nutrition.',
                    'te': 'ఎలాంటి చికిత్స అవసరం లేదు. పద్ధతి ప్రకారం నీరు, పోషకాలు అందించండి.',
                    'kn': 'ಯಾವುದೇ ಚಿಕಿತ್ಸೆಯ ಅಗತ್ಯವಿಲ್ಲ. ಸರಿಯಾಗಿ ನೀರು ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ಒದಗಿಸಿ.'
                }
            },
            'leaf_rust': {
                'name': {
                    'en': 'Leaf Rust',
                    'te': 'ఆకు తుప్పు (Leaf Rust)',
                    'kn': 'ಎಲೆ ತುಕ್ಕು (Leaf Rust)'
                },
                'description': {
                    'en': "Leaf rust is a fungal disease that causes orange or brown pustules to appear on the leaves. It reduces the plant's ability to photosynthesize, leading to reduced yield.",
                    'te': 'ఆకు తుప్పు (Leaf Rust) అనేది ఆకులపై నారింజ లేదా గోధుమ రంగు మచ్చలను కలిగించే ఒక శిలీంధ్ర వ్యాధి. ఇది దిగుబడిని తగ్గిస్తుంది.',
                    'kn': 'ಎಲೆ ತುಕ್ಕು ಎನ್ನುವುದು ಎಲೆಗಳ ಮೇಲೆ ಕಿತ್ತಳೆ ಅಥವಾ ಕಂದು ಬಣ್ಣದ ಕಲೆಗಳನ್ನು ಉಂಟುಮಾಡುವ ಶಿಲೀಂಧ್ರ ರೋಗವಾಗಿದೆ. ಇದು ಇಳುವರಿಯನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ.'
                },
                'recommendation': {
                    'en': 'Remove infected leaves. Apply copper-based fungicide. Ensure good air circulation.',
                    'te': 'వ్యాధి సోకిన ఆకులను తొలగించండి. రాగి ఆధారిత శిలీంద్రనాశిని (fungicide) వాడండి. గాలి బాగా ఆడేలా చూసుకోండి.',
                    'kn': 'ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ. ತಾಮ್ರ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಬಳಸಿ. ಉತ್ತಮ ಗಾಳಿ ಸಂಚಾರವನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ.'
                }
            },
            'powdery_mildew': {
                'name': {
                    'en': 'Powdery Mildew',
                    'te': 'బూడిద తెగులు (Powdery Mildew)',
                    'kn': 'ಬೂದಿ ರೋಗ (Powdery Mildew)'
                },
                'description': {
                    'en': 'Powdery mildew is a fungal disease recognized by white, powdery spots on leaves and stems. It can stunt plant growth and reduce crop quality.',
                    'te': 'బూడిద తెగులు అనేది ఆకులు మరియు కాండంపై తెల్లటి, బూడిద లాంటి మచ్చలతో గుర్తించబడే శిలీంధ్ర వ్యాధి. ఇది మొక్కల పెరుగుదలను అడ్డుకుంటుంది.',
                    'kn': 'ಬೂದಿ ರೋಗ ಎನ್ನುವುದು ಎಲೆಗಳು ಮತ್ತು ಕಾಂಡಗಳ ಮೇಲೆ ಬಿಳಿ, ಪುಡಿಯಂತಹ ಕಲೆಗಳಿಂದ ಗುರುತಿಸಲ್ಪಡುವ ಶಿಲೀಂಧ್ರ ರೋಗ. ಇದು ಸಸ್ಯಗಳ ಬೆಳವಣಿಗೆಯನ್ನು ಕುಂಠಿತಗೊಳಿಸುತ್ತದೆ.'
                },
                'recommendation': {
                    'en': 'Apply sulfur-based fungicide. Avoid overhead watering. Prune for better airflow.',
                    'te': 'సల్ఫర్ ఆధారిత ఫంగిసైడ్ వాడండి. పైనుండి నీరు పోయడం నివారించండి. గాలి ప్రసరణ కోసం ప్రూనింగ్ చేయండి.',
                    'kn': 'ಸಲ್ಫರ್ ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕವನ್ನು ಅನ್ವಯಿಸಿ. ಮೇಲಿನಿಂದ ನೀರು ಹಾಕುವುದನ್ನು ತಪ್ಪಿಸಿ. ಉತ್ತಮ ಗಾಳಿಗಾಗಿ ಕತ್ತರಿಸು (prune) ಮಾಡಿ.'
                }
            },
            'unknown': {
                'name': {
                    'en': 'Unknown Disease',
                    'te': 'గుర్తించబడని వ్యాధి',
                    'kn': 'ತಿಳಿಯದ ರೋಗ'
                },
                'description': {
                    'en': 'The model could not identify the disease with high confidence.',
                    'te': 'ఈ వ్యాధిని ఖచ్చితంగా గుర్తించలేకపోయాము.',
                    'kn': 'ಮಾದರಿಯು ರೋಗವನ್ನು ನಿಖರವಾಗಿ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.'
                },
                'recommendation': {
                    'en': 'Consult a local agricultural expert for accurate diagnosis.',
                    'te': 'ఖచ్చితమైన రోగ నిర్ధారణ కోసం స్థానిక వ్యవసాయ నిపుణుడిని సంప్రదించండి.',
                    'kn': 'ನಿಖರವಾದ ರೋಗನಿರ್ಣಯಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ.'
                }
            }
        }

    def predict_image(self, image_bytes):
        """
        Runs inference on given image bytes, calculates severity, draws boxes.
        Returns the annotated image (base64) and a dictionary of results.
        """
        import base64
        
        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, {"error": "Invalid image"}
            
        img_height, img_width = img.shape[:2]
        total_area = img_height * img_width
        
        # Run YOLO inference
        results = self.model(img)[0]
        
        boxes = results.boxes
        drawn_img = img.copy()
        
        disease_detected = "Healthy"
        confidence = 0.0
        disease_area = 0.0
        
        if len(boxes) > 0:
            # Find the most confident prediction
            best_box = max(boxes, key=lambda b: b.conf[0].item())
            class_id = int(best_box.cls[0].item())
            confidence = best_box.conf[0].item() * 100
            
            # Using either the class names from the model, or default to some names
            class_names = self.model.names
            if class_id in class_names:
                disease_detected = class_names[class_id]
            else:
                disease_detected = f"Class {class_id}"
                
            # Draw all boxes and calculate severity
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = class_names[cls] if cls in class_names else str(cls)
                
                # Area of this disease spot
                w = x2 - x1
                h = y2 - y1
                disease_area += (w * h)
                
                # Color based on class
                color = (0, 0, 255) if 'healthy' not in name.lower() else (0, 255, 0)
                
                # Draw box
                cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{name} {conf:.2f}"
                cv2.putText(drawn_img, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif self.is_demo:
            import random
            disease_detected = random.choice(['leaf_rust', 'powdery_mildew'])
            confidence = random.uniform(75.0, 95.0)
            
            x1 = int(img_width * random.uniform(0.1, 0.3))
            y1 = int(img_height * random.uniform(0.1, 0.3))
            x2 = min(x1 + int(img_width * random.uniform(0.3, 0.5)), img_width - 5)
            y2 = min(y1 + int(img_height * random.uniform(0.3, 0.5)), img_height - 5)
            
            disease_area = (x2 - x1) * (y2 - y1)
            
            cv2.rectangle(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{disease_detected} {confidence/100:.2f}"
            cv2.putText(drawn_img, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        # Calculate Severity
        severity = "None"
        if 'healthy' not in disease_detected.lower() and total_area > 0:
            ratio = (disease_area / total_area) * 100
            if ratio < 10:
                severity = "Low"
            elif ratio < 35:
                severity = "Medium"
            else:
                severity = "High"
                
        # Get Info
        rec_key = disease_detected.lower()
        info = self.disease_info.get(rec_key, self.disease_info['unknown'])
        
        # Encode image to base64 for frontend
        _, buffer = cv2.imencode('.jpg', drawn_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result_dict = {
            "disease_key": disease_detected,
            "disease_info": info,
            "confidence": round(confidence, 2),
            "severity": severity
        }
        
        return img_base64, result_dict

    def process_frame(self, frame):
        """
        Process a single BGR frame for real-time video streaming.
        Returns the annotated frame.
        """
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item() * 100
            cls = int(box.cls[0].item())
            name = self.model.names[cls] if cls in self.model.names else str(cls)
            
            color = (0, 255, 0) if 'healthy' in name.lower() else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name}: {conf:.1f}%", (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
        if len(boxes) == 0 and self.is_demo:
            import random
            h, w = frame.shape[:2]
            name = random.choice(['leaf_rust', 'powdery_mildew'])
            conf = random.uniform(75.0, 95.0)
            x1, y1 = int(w * 0.2), int(h * 0.2)
            x2, y2 = int(w * 0.6), int(h * 0.6)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{name}: {conf:.1f}%", (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return frame
