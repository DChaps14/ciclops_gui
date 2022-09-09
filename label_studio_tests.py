import webbrowser, os
import shutil
import time
import json
import numpy as np
from PIL import Image, ImageDraw
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
        point = (round(point[0]*image_width*0.01), round(point[1]*image_height*0.01))
        points[index] = point
        if point[0] < minx:
            minx = point[0]
        elif point[0] > maxx:
            maxx = point[0]
        if point[1] < miny:
            miny = point[1]
        elif point[1] > maxy:
            maxy = point[1]
    
    drawing_image = Image.new("L", (image_width, image_height), 0)
    polygon_draw = ImageDraw.Draw(drawing_image)
    polygon_draw.polygon(points, fill=1, outline=1)
    full_mask = np.array(drawing_image)
    full_mask = np.resize(full_mask, (image_height, image_width, 1))
    full_mask = np.where(full_mask, label_int, 0)
    full_image_mask = np.where(full_mask, label_int, full_image_mask)
    crop_image = image_array[miny:maxy, minx:maxx]
    crop_mask = full_mask[miny:maxy, minx:maxx]
    plt.imshow(tf.keras.utils.array_to_img(crop_image))
    plt.imshow(tf.keras.utils.array_to_img(crop_mask), alpha=0.4)    
    plt.show()  
    detection = {"label": label, "mask": np.ndarray.tolist(crop_mask),
                 "bounding_box": [minx, miny, maxx, maxy]}
    detections.append(detection)
    
plt.imshow(tf.keras.utils.array_to_img(image_array))
plt.imshow(tf.keras.utils.array_to_img(full_image_mask), alpha=0.4)
plt.show()
filename = str(abs(hash(time.localtime())))
image.save(f"UNetPredictions/usableImages/{filename}.jpg")
mask_json = {"filename": filename+".jpg", "ground_truth": {"detections": detections},
        "skip_full_mask": False}
with open(f"UNetPredictions/usableImages/{filename}.json", 'w') as mask_file:
    json.dump(mask_json, mask_file)    
    
    
            
        #last_point = [round(points[index-1][0]*image_width*0.01), round(points[index-1][1]*image_height*0.01)]
        #if index == len(points)-1:
            #next_index = 0
        #else:
            #next_index = index+1
        #next_point_y = round(points[next_index][1]*image_height*0.01)
        
        ## If a point is higher or lower than both of its neighbours, its a peak
        #is_peak = (point[1] < last_point[1] and point[1] < next_point_y) or (point[1] > last_point[1] and point[1] > next_point_y)
          
        #x_vals = mask_points.get(point[1])
        #if x_vals == None:
            #peak_list = ([], [point[0]])[is_peak]
            #slope_list = ([point[0]], [])[is_peak]
            #mask_points[point[1]] = {"peaks":peak_list, "slopes": slope_list}
        #else:
            #key = ("slopes", "peaks")[is_peak]
            #x_vals[key] += [point[0]]
            #mask_points[point[1]] = x_vals
        
        
        #if point[0] == last_point[0]:
            
            #for y_point in range(min(point[1], last_point[1])+1, max(point[1], last_point[1])):
                #x_vals = mask_points.get(y_point)
                #if x_vals:
                    #x_vals["slopes"] += [point[0]]
                    #mask_points[y_point] = x_vals
                #else:
                    #mask_points[y_point] = {"peaks": [], "slopes": [point[0]]}
    
        ## Should theoretically already be taken care of
        ##elif point[1] == last_point[1]:
            ##min_point = min(point[0], last_point[0])
            ##max_point
            
            ##for x_point in range(min(point[0], last_point[0])+1, max(point[0], last_point[0])):
                ##x_vals = mask_points.get(point[1])
                ##if x_vals:
                    ##mask_points[point[1]] = x_vals + [x_point]
                ##else:
                    ##mask_points[point[1]] = [x_point]            
        
        #else:
            #gradient = (point[1] - last_point[1]) / (point[0] - last_point[0])
            #intercept = point[1] - point[0]*gradient     
            
            ##for x_point in range(min(point[0], last_point[0])+1, max(point[0], last_point[0])):
                ##y_point = round((gradient*x_point) + intercept)
                ##x_vals = mask_points.get(y_point)
                ##if x_vals:
                    ##x_vals["slopes"] += [x_point]
                    ##mask_points[y_point] = x_vals
                ##else:
                    ##mask_points[y_point] = {"slopes": [x_point], "peaks": []}
            #for y_point in range(min(point[1], last_point[1])+1, max(point[1], last_point[1])):
                #x_point = round((y_point - intercept)/gradient)
                #x_vals = mask_points.get(y_point)
                #if x_vals and x_point not in x_vals["slopes"]:
                    #x_vals["slopes"] += [x_point]
                    #mask_points[y_point] = x_vals
                #elif not x_vals:
                    #mask_points[y_point] = {"slopes": [x_point], "peaks": []}
    ## mask_points should now contain at most 2 points on each y value of the image
    ## If there are two points, the mask exists between these two points on the image
    ## If there is one point, only that point is part of the mask
    ## If there are no points, that image row is not part of the mask - this will also not a part of the dictionary
    #crop_image = image_array[miny:maxy, minx:maxx]
    #for val in mask_points.items():
        #y_val = val[0]
        #x_vals = val[1]
        #x_slps = sorted(x_vals.get("slopes"))
        #x_pks = x_vals.get("peaks")
        #if len(x_slps) == 1:
            #x_pks = x_slps
            #x_slps = []
        #elif len(x_slps) % 2 == 1:            
            #prev_y_peak = mask_points[y_val-1].get("peaks")
            #next_y_peak = mask_points[y_val+1].get("peaks")
            #if prev_y_peak:
                #x_slps.remove(prev_y_peak)
            #elif next_y_peak:
                #x_slps.remove(next_y_peak)
        #print(y_val, x_pks, x_slps)
        
        #if x_pks:
            #for peak in x_pks:
                #full_image_mask[y_val-1][peak][0] = label_int
        #if x_slps:
            #for index in range(0, len(x_slps), 2):
                    #min_val = min(x_slps[index], x_slps[index+1])
                    #max_val = max(x_slps[index], x_slps[index+1])
                    #for pos in range(min_val, max_val+1):
                        #full_image_mask[y_val-1][pos][0] = label_int
            
        
        ##for x in x_slps:
            ##full_image_mask[y_val][x][0] = label_int
        ##for x in x_pks:
            ##full_image_mask[y_val][x][0] = label_int
        ##if len(x_vals) == 1:
            ##full_image_mask[y_val][x_vals[0]][0] = label_int
        ##else:
            ##for index in range(min(x_vals), max(x_vals)+1):
                ##full_image_mask[y_val][index][0] = label_int        
    #crop_mask = full_image_mask[miny:maxy, minx:maxx]
    #crop_mask = np.where(crop_mask==label_int, label_int, 0)
    #plt.imshow(tf.keras.utils.array_to_img(crop_image))
    #plt.imshow(tf.keras.utils.array_to_img(crop_mask), alpha=0.4)    
    #plt.show()
    #detection = {"label": label, "mask": np.ndarray.tolist(crop_mask),
                 #"bounding_box": [minx, miny, maxx, maxy]}
    #detections.append(detection)
    
#plt.imshow(tf.keras.utils.array_to_img(image_array))
#plt.imshow(tf.keras.utils.array_to_img(full_image_mask), alpha=0.4)
#plt.show()
#filename = str(hash(time.localtime()))
#image.save(f"UNetPredictions/usableImages/{filename}.jpg")
#mask_json = {"filename": filename+".jpg", "ground_truth": {"detections": detections},
        #"skip_full_mask": False}
#with open(f"UNetPredictions/usableImages/{filename}.json", 'w') as mask_file:
    #json.dump(mask_json, mask_file)