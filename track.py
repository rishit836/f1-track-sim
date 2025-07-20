import cv2
import numpy as np
import os
import cairosvg
import matplotlib.pyplot as plt
import sys
import math
# using skeletonize to convert the track into a singular path
from skimage.morphology import skeletonize


def convert_svg(track_name:str)->bool:
    cairosvg.svg2png(url=f"tracks/svg/{track_name}.svg",write_to=f"tracks/png/{track_name}.png")
    return True

def convert_track(track_name,verbose:bool="True"):
    if not os.path.exists(f"tracks/png/{track_name}.png"):
        if verbose:
            print(f"{track_name.upper()} PNG not avaiable checking for svg..")
        if not os.path.exists(f"tracks/svg/{track_name}.svg"):
            if verbose:
                print(f"{track_name.upper()} SVG also not available please provide the svg or double check the name")
            return False
        else:
            if verbose:
                print(f"{track_name.upper()} SVG file found")
            converted_=convert_svg(track_name)
            if converted_:
                print(f"{track_name.upper()} PNG file created.")
    else:
        if verbose:
            print(f"{track_name.upper()} File Already Exists.")
    img_path = f"tracks/png/{track_name}.png"
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    b,g,r,a = cv2.split(img)
    thresh = 10

    # checking if the pixel is black
    black_pixels = (r<thresh) & (g<thresh) &(b<thresh)

    # checking if the pixel is visible
    visible = (a>0)
    mask = black_pixels & visible
    # converting the picture into a path

    mk = skeletonize(mask)

    # stacking two arrays into 2d array (x,y)
    black_coords = np.column_stack(np.where(mask))
    black_coords_1 = np.column_stack(np.where(mk))
    black_coords_1 = sort_nearest_neighbour(black_coords_1)


    track_path = [(int(x), int(y)) for y, x in black_coords]
    track_path1 = [(int(x), int(y)) for y, x in black_coords_1]
    if verbose:
        print(verbose)
        x_vals = [pt[0] for pt in track_path]
        y_vals = [pt[1] for pt in track_path]
        x_vals1 = [pt[0] for pt in track_path1]
        y_vals1 = [pt[1] for pt in track_path1]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, s=1, c='black')
        plt.scatter(x_vals1, y_vals1, s=1, c='blue')

        # Plot the first track point in red (larger marker)
        plt.scatter(x_vals1[0], y_vals1[0], s=30, c="red", label="Start Point")
        plt.gca().invert_yaxis()  # Match image coordinate system
        plt.title(f"{track_name.upper()} (Black Pixels on Transparent PNG)")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return track_path1

def sort_nearest_neighbour(path):
    # creating a path to be used
    used = [False] * len(path)
    curr_index = 0
    used[curr_index] = True
    sorted_path = []
    while not all(used):
        dist = 0
        nearest_index = -1
        nearest_distance = float('inf')
        for index,point in enumerate(path):
            # skippings points which are already sorted/used
            if used[index]:
                continue
            dist = math.dist(path[curr_index],point)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_index = index
        curr_index = nearest_index
        used[curr_index] =True
        sorted_path.append(path[curr_index])
        
    return sorted_path

def are_collinear(points):
    if len(points)<3:
        return True
    # because we require slopes to set car angle
    slopes = []
    for index,point in enumerate(points):
        for i,point2 in enumerate(points):
            if i == index:
                continue
            y_diff = point2[1] - point[1]
            x_diff = point2[0] - point[0]
            if x_diff != 0:
                slope = y_diff/x_diff
                slopes.append(slope)
            else:
                slopes.append(float('inf'))
    if len(slopes) !=0:
        mean_slope = sum(slopes)/len(slopes)
    else:
        mean_slope= float('inf')
    for s in slopes:
        # having a error window to allow the slope to be almost equal
        #or if a track is a little bit weird
        if not abs(s-mean_slope) < math.tan(math.radians(40)):
            # if any point has bigger error difference than 1e-3 then
            # we return False
            # else the loop keeps running and return True
            return False
    
    return True
            

# creating sectors such as straight line curve with curve radius etc and not purple green sectors
# these sectors can then be used to generate a optimal racing line which can be used by the
# model to learn
def create_sector(path,window_size:int=3):
    for index,point in enumerate(path):
        if index+window_size<len(path):
            window = path[index:window_size+index]
        else:
            window = path[index:]
        if len(window) == 3:
            window = np.array(window)
            p1 = window[0]
            p2 = window[1]
            p3 = window[2]

            # building vectors
            v1 = p2-p1
            v2 = p3-p2

            dot = np.sum(v2*v1,axis=1)
            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)

            # Avoid division by zero
            norm_prod = norm1 * norm2
            norm_prod[norm_prod == 0] = 1e-8

            # Compute the cosine of the angle
            cos_theta = dot / norm_prod

            # Clip values to valid range to avoid NaNs due to numerical precision
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            angles = np.arccos(cos_theta)   # In radians
            angles_deg = np.degrees(angles) # In degrees if you prefer
            
            



def create_track(track_name:str="all",verbose:bool=True):
    if not track_name.lower()  == "all":
        t = convert_track(track_name=track_name,verbose=verbose)
        return t
    else:
        track_paths={}
        for filename in os.listdir("tracks/svg"):
            full_path = os.path.join("tracks",filename)
            t = convert_track(track_name=filename.split(".")[0],verbose=verbose)
            track_paths.update({filename.split(".")[0]:t})

        return track_paths



if __name__ == "__main__":
    try:
        name = sys.argv[1]
        print(name)
    except:
        name="all"  

    track = create_track(name)


