import cv2
import numpy as np
import json
import os

def analyze_metal(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
    brightness = np.mean(img)
    contrast = np.std(img)
    edges = cv2.Canny(img, 100, 200)
    grain_density = np.sum(edges > 0) / img.size
    return [float(brightness), float(contrast), float(grain_density)]

# Load labels
with open('label.json', 'r') as f:
    data = json.load(f)

metal_data = {}
photos_dir = r'C:\Programming_Project\Micro_learn' # TRIPLE CHECK THIS FOLDER NAME

print(f"Starting to process photos in {photos_dir}...")

success_count = 0

for entry in data:
    try:
        label = entry['annotations'][0]['result'][0]['value']['choices'][0]
        
        filename = entry['file_upload']
        
        path = os.path.join(photos_dir, filename)
        
        features = analyze_metal(path)
        
        if features:
            if label not in metal_data:
                metal_data[label] = []
            metal_data[label].append(features)
            success_count += 1
            if success_count % 50 == 0:
                print(f"processing {success_count} photos...")
    except Exception as e:
        continue

if success_count == 0:
    print("fail. check folder name and image name")
else:
    final_library = {}
    for metal, values in metal_data.items():
        averages = np.mean(values, axis=0)
        final_library[metal] = averages.tolist()
        print(f"Calculated average for: {metal}")

    with open('metal_library.json', 'w') as f:
        json.dump(final_library, f)
    print(f"\nlibrary built with {success_count} photos.")