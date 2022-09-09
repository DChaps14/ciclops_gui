import webbrowser, os
import shutil
import time
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

class_dict = {"dog":1, "cat":2}

# Open the label_studio 
webbrowser.open('file://' + os.path.realpath("label_studio.html"))

current_path = os.path.dirname(os.path.abspath(__file__))
path = current_path.split("\\")[:3]
download_path = "\\".join(path) + "\\Downloads\\annotation.txt"
while not os.path.exists(download_path):
    time.sleep(5)
    print("waiting")
    
shutil.copy(download_path, "./newAnnotation.txt")
os.remove(download_path)


# Need to read from the newAnnotation file and extract the information from it. 
# Then need to find the bounding box of each annotation, and create a mask within this bbox. After this, we can remove it from the directory, and move it into usable images

image = Image.open("chosen_image.jpg")
image_array = np.array(image)
image_height, image_width, _ = image_array.shape
full_image_mask = np.zeros((len(image_array), len(image_array[0]), 1))
detections = []
annotation_file = open("newAnnotation.txt")
annotation_info = annotation_file.readlines()
for annot_json in annotation_info:
    annot = json.loads(annot_json)
    points = annot["value"]["points"]
    label = annot["value"]["polygonlabels"][0]
    label_int = class_dict.get(label)
    mask_points = {}
    minx, miny, maxx, maxy = round(points[0][0]*image_width*0.01), round(points[0][1]*image_height*0.01), round(points[0][0]*image_width*0.01), round(points[0][1]*image_height*0.01)
    for index, point in enumerate(points):
        point = [round(point[0]*image_width*0.01), round(point[1]*image_height*0.01)]
        if point[0] < minx:
            minx = point[0]
        elif point[0] > maxx:
            maxx = point[0]
        if point[1] < miny:
            miny = point[1]
        elif point[1] > maxy:
            maxy = point[1]
          
        x_vals = mask_points.get(point[1])
        if x_vals == None:
            mask_points[point[1]] = [point[0]]
        else:
            mask_points[point[1]] = x_vals + [point[0]]
        
        last_point = [round(points[index-1][0]*image_width*0.01), round(points[index-1][1]*image_height*0.01)]
        if point[0] == last_point[0]:
            gradient = "vert"
        elif point[1] == last_point[1]:
            gradient = "horz"
        else:
            gradient = (point[1] - last_point[1]) / (point[0] - last_point[0])
            intercept = point[1] - point[0]*gradient
        if gradient == "vert":
            for y_point in range(min(point[1], last_point[1])+1, max(point[1], last_point[1])):
                x_vals = mask_points.get(y_point)
                if x_vals:
                    mask_points[y_point] = x_vals + [point[0]]
                else:
                    mask_points[y_point] = [point[0]]       
        elif gradient == "horz":
            for x_point in range(min(point[0], last_point[0])+1, max(point[0], last_point[0])):
                x_vals = mask_points.get(point[1])
                if x_vals:
                    mask_points[point[1]] = x_vals + [x_point]
                else:
                    mask_points[point[1]] = [x_point]
        else:           
            for x_point in range(min(point[0], last_point[0])+1, max(point[0], last_point[0])):
                y_point = round((gradient*x_point) + intercept)
                x_vals = mask_points.get(y_point)
                if x_vals:
                    mask_points[y_point] = x_vals + [x_point]
                else:
                    mask_points[y_point] = [x_point]
            for y_point in range(min(point[1], last_point[1])+1, max(point[1], last_point[1])):
                x_point = round((y_point - intercept)/gradient)
                x_vals = mask_points.get(y_point)
                if x_vals and x_point not in x_vals:
                    mask_points[y_point] = x_vals + [x_point]
                elif not x_vals:
                    mask_points[y_point] = [x_point]
    # mask_points should now contain at most 2 points on each y value of the image
    # If there are two points, the mask exists between these two points on the image
    # If there is one point, only that point is part of the mask
    # If there are no points, that image row is not part of the mask - this will also not a part of the dictionary
    crop_image = image_array[miny:maxy, minx:maxx]
    for val in mask_points.items():
        y_val = val[0]
        x_vals = val[1]
        if len(x_vals) == 1:
            full_image_mask[y_val][x_vals[0]][0] = label_int
        else:
            for index in range(min(x_vals), max(x_vals)+1):
                full_image_mask[y_val][index][0] = label_int
    crop_mask = full_image_mask[miny:maxy, minx:maxx]
    crop_mask = np.where(crop_mask==label_int, label_int, 0)
    plt.imshow(tf.keras.utils.array_to_img(crop_image))
    plt.imshow(tf.keras.utils.array_to_img(crop_mask), alpha=0.4)    
    plt.show()
    detection = {"label": label, "mask": np.ndarray.tolist(crop_mask),
                 "bounding_box": [minx, miny, maxx, maxy]}
    detections.append(detection)
    
plt.imshow(tf.keras.utils.array_to_img(image_array))
plt.imshow(tf.keras.utils.array_to_img(full_image_mask), alpha=0.4)
plt.show()
filename = str(hash(time.localtime()))
image.save(f"UNetPredictions/usableImages/{filename}.jpg")
mask_json = {"filename": filename+".jpg", "ground_truth": {"detections": detections},
        "skip_full_mask": False}
with open(f"UNetPredictions/usableImages/{filename}.json", 'w') as mask_file:
    json.dump(mask_json, mask_file)