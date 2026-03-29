import os
import csv
from datetime import datetime

HISTORY_FILE = "history.csv"

def log_prediction(disease, confidence, severity):
    """
    Logs a prediction result to a CSV file.
    """
    file_exists = os.path.isfile(HISTORY_FILE)
    
    with open(HISTORY_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Disease Detected", "Confidence (%)", "Severity"])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, disease, confidence, severity])

def get_history():
    """Reads the history from the CSV file."""
    if not os.path.exists(HISTORY_FILE):
        return []
        
    history = []
    with open(HISTORY_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(row)
    return history
