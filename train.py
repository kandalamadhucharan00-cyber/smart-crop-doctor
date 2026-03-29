import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def setup_yolo_dataset(base_dir, yolo_dir):
    """
    Converts a classification dataset (subfolders) into YOLO object detection format.
    Generates dummy bounding boxes assuming the object (leaf) is centered.
    """
    print("Preparing YOLO dataset format...")
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    classes.sort()
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, 'labels', split), exist_ok=True)
        
    all_images = []
    all_labels = []
    
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls_name)
        images = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            all_images.append(img_path)
            all_labels.append((img_name, cls_idx))
            
    # Split into train/val
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )
    
    def process_split(img_paths, labels, split):
        for img_path, (img_name, cls_idx) in zip(img_paths, labels):
            # Copy image
            dst_img = os.path.join(yolo_dir, 'images', split, img_name)
            shutil.copy(img_path, dst_img)
            
            # Create dummy label (Bounding box: x_center, y_center, width, height)
            # Assuming the leaf is in the center and occupies ~60-80% of the image.
            label_name = os.path.splitext(img_name)[0] + '.txt'
            dst_label = os.path.join(yolo_dir, 'labels', split, label_name)
            with open(dst_label, 'w') as f:
                # Format: <class> <x_center> <y_center> <width> <height>
                f.write(f"{cls_idx} 0.5 0.5 0.7 0.7\n")
                
    process_split(train_imgs, train_labels, 'train')
    process_split(val_imgs, val_labels, 'val')
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(yolo_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"Dataset prepared at {yolo_dir} with {len(classes)} classes: {classes}")
    return yaml_path

def train_model(data_yaml):
    print("Starting YOLOv8 training. This may take some time depending on your hardware...")
    
    # Load a pretrained YOLOv8 Nano model (best for fast training/real-time)
    model = YOLO('yolov8n.pt') 
    
    # Train the model
    # Note: We use 10 epochs for demo purposes. Increase 'epochs' for better accuracy.
    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=256,
        batch=8,
        name='crop_doctor_model',
        project='runs/detect'
    )
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Copy best model to our standard models folder
    best_model_path = os.path.join('runs', 'detect', 'crop_doctor_model', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, os.path.join('models', 'best.pt'))
        print("\nSUCCESS! Model trained and saved to models/best.pt")
    else:
        print("\nWARNING: Could not find best.pt. Training may have failed.")

def main():
    base_dataset = r"c:\Users\kanda\agriculture\agriculture"
    yolo_dataset = r"c:\Users\kanda\agriculture\yolo_dataset"
    
    if not os.path.exists(base_dataset):
        print(f"Error: Base dataset directory NOT FOUND at {base_dataset}!")
        print("Please run 'python generate_dummy_data.py' first to generate synthetic data, or ensure your images are placed correctly.")
        return
        
    yaml_path = setup_yolo_dataset(base_dataset, yolo_dataset)
    train_model(yaml_path)

if __name__ == "__main__":
    main()
