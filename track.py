import cv2
import numpy as np
import os
import cairosvg
import matplotlib.pyplot as plt
import sys
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


