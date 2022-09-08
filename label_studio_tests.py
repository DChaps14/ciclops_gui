import webbrowser, os
import shutil
import time

# Open the label_studio 
webbrowser.open('file://' + os.path.realpath("label_studio.html"))

#current_path = os.path.dirname(os.path.abspath(__file__))
#path = current_path.split("\\")[:3]
#download_path = "\\".join(path) + "\\Downloads\\annotation.txt"
#while not os.path.exists(download_path):
    #time.sleep(10)
    #print("waiting")
    
#shutil.copy(download_path, "./newAnnotation.txt")
#os.remove(download_path)


# Need to read from the newAnnotation file and extract the information from it. 
# Then need to find the bounding box of each annotation, and create a mask within this bbox. After this, we can remove it from the directory, and move it into usable images