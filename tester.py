
import cv2
import numpy as np
import json
import os

# load the library we just built
with open('metal_library.json', 'r') as f:
    library = json.load(f)

def analyze_metal(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    brightness = np.mean(img)
    contrast = np.std(img)
    edges = cv2.Canny(img, 100, 200)
    grain_density = np.sum(edges > 0) / img.size
    return np.array([brightness, contrast, grain_density])

def identify_metal(test_path):
    test_features = analyze_metal(test_path)
    if test_features is None:
        return "File not found!"

    best_match = None
    min_distance = float('inf')

    # compare the test photo to every metal in our library
    for metal_name, library_averages in library.items():
        library_averages = np.array(library_averages)
        
        # Calculate the "Distance" between the numbers
        distance = np.linalg.norm(test_features - library_averages)
        
        if distance < min_distance:
            min_distance = distance
            best_match = metal_name
            
    return best_match, min_distance

# run test

test_folder = r'C:\Programming_Project\Micro_test'

#folder check
if not os.path.exists(test_folder):
    print(f"can't find the folder at {test_folder}")
else:
    files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#photo check
    if len(files) == 0:
        print(f"folder found, no photos in {test_folder}")
    else:
        print(f"found {len(files)} test photos. starting analysis...\n")
        print(f"{'Filename':<25} | {'Predicted Metal':<15} | {'Distance'}")
        print("-" * 60)

        for filename in files:
            path = os.path.join(test_folder, filename)
            
            # call the function and get the two returns
            prediction, dist = identify_metal(path)
            
            # printing the actual results
            print(f"{filename:<25} | {prediction:<15} | {dist:.2f}")

print("\n--- Testing Complete ---")