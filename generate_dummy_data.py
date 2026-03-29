import os
import cv2
import numpy as np

def create_leaf_image(disease_type, img_size=(256, 256)):
    # Create dark background
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    # Draw a green leaf shape (ellipse)
    center = (img_size[0]//2, img_size[1]//2)
    axes = (80, 110)
    angle = np.random.randint(-20, 20)
    cv2.ellipse(img, center, axes, angle, 0, 360, (50, 200, 50), -1)
    
    # Add disease features
    if disease_type == 'leaf_rust':
        # Add brown/orange spots
        for _ in range(np.random.randint(5, 15)):
            x = np.random.randint(center[0]-50, center[0]+50)
            y = np.random.randint(center[1]-80, center[1]+80)
            r = np.random.randint(3, 10)
            cv2.circle(img, (x, y), r, (0, 100, 200), -1) # BGR
            
    elif disease_type == 'powdery_mildew':
        # Add white/gray patches
        for _ in range(np.random.randint(3, 8)):
            x = np.random.randint(center[0]-60, center[0]+60)
            y = np.random.randint(center[1]-90, center[1]+90)
            r = np.random.randint(10, 25)
            cv2.circle(img, (x, y), r, (220, 220, 220), -1)
            
    # 'healthy' has no spots
    return img

def main():
    base_dir = r"c:\Users\kanda\agriculture\agriculture"
    classes = ['healthy', 'leaf_rust', 'powdery_mildew']
    num_samples = 30 # enough for a quick YOLO training
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(num_samples):
            img = create_leaf_image(cls)
            # Add some slight noise
            noise = np.random.normal(0, 5, img.shape).astype(np.int8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            filepath = os.path.join(cls_dir, f"{cls}_{i}.jpg")
            cv2.imwrite(filepath, img)
            print(f"Generated {filepath}")
            
    print("Dummy classification dataset generation complete!")

if __name__ == "__main__":
    main()
