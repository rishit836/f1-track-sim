import cv2
import numpy as np
import os
import cairosvg
import matplotlib.pyplot as plt
import sys
import math
# using skeletonize to convert the track into a singular path
from skimage.morphology import skeletonize
from scipy.interpolate import splprep,splev


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

    track_path1 = smooth_path(track_path1,len(track_path1)*.05)
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

def smooth_path(path,smoothness:int=0):
    # unpack the x,y into variablkes
    x,y = np.array(path).T

    # B-spline curve to smooth out the path from N-D curve
    # per=True for closed tracks
    tck,u = splprep([x,y],s=smoothness, per=True)

    # creating more points for a smoother curve
    # points are created on the spline curve
    u_new = np.linspace(u.min(),u.max(),num=len(path)*2)

    # evaluate the new spline curve created
    x_new,y_new = splev(u_new,tck)
    smoothed_path = list(zip(x_new,y_new))
    return smoothed_path


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
def set_used(use_map:list,start:int,end:int)->list:
    for i,v in enumerate(use_map):
        if start<=i and end>=i:
            use_map[i] = True
        if i > end:
            break
    return use_map

def create_sectors(path,window_size:int=11):
    use_map = [False]*len(path)
    sector_map = {
        'straight':[],
        'turn':[]
    }
    for i,p in enumerate(path):
        # just to make sure the points have not already been used
        if not use_map[i]:
            # creating window
            if i+window_size<len(path):
                start = i
                end = i+window_size+1
                window = path[start:end]
                use_map = set_used(use_map,start,end)
            else:
                start = i
                end = len(path)-1
                window = path[start:]
                use_map = set_used(use_map,start,end)
                
            # center point of the window
            mid_ = len(window) // 2
            w= np.array(window)
            # performing vector operations
            done = False
            pointer = 0
            pd = []
            while not done:
                if pointer >= len(w):
                    done=True
                    break
                v1 = w[pointer:pointer+1]
                v2 = w[pointer+1:pointer+2]
                v1_mag = np.linalg.norm(v1)
                v2_mag = np.linalg.norm(v2)
                cross_product = np.sum(np.cross(v1,v2))
                angle = math.degrees(math.acos(cross_product/(v1_mag*v2_mag)))
                pd.append(angle)
                pointer+=1
            for index,angle in enumerate(pd):
                if 89 <= angle<90:
                    if window[index] not in sector_map['turn']:
                        sector_map['straight'].append(window[index])
                else:
                    if window[index] not in sector_map['straight']:
                        sector_map['turn'].append(window[index])
    return sector_map
        

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


