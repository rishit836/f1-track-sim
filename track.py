import cv2
import numpy as np
import os
import cairosvg
import matplotlib.pyplot as plt


def convert_svg(track_name:str)->bool:
    cairosvg.svg2png(url=f"tracks/svg/{track_name}.svg",write_to=f"tracks/png/{track_name}.png")
    return True

def create_track(track_name:str="miami",verbose:bool=True):
    if not os.path.exists(f"tracks/png/{track_name}.png"):
        if verbose:
            print("PNG not avaiable checking for svg..")
        if not os.path.exists(f"tracks/svg/{track_name}.png"):
            if verbose:
                print("SVG also not available please provide the svg or double check the name")
            return False
        else:
            if verbose:
                print("SVG file found")
            converted_=convert_svg(track_name)
            if converted_:
                print("PNG file created.")
    else:
        if verbose:
            print("File Already Exists.")
    img_path = f"tracks/png/{track_name}.png"
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    b,g,r,a = cv2.split(img)
    thresh = 10

    # checking if the pixel is black
    black_pixels = (r<thresh) & (g<thresh) &(b<thresh)

    # checking if the pixel is visible
    visible = (a>0)
    mask = black_pixels & visible

    # stacking two arrays into 2d array (x,y)
    black_coords = np.column_stack(np.where(mask))
    track_path = [(int(x), int(y)) for y, x in black_coords]
    
    return track_path


if __name__ == "__main__":

    track = create_track("miami")
    if track:
        
        x_vals = [pt[0] for pt in track]
        y_vals = [pt[1] for pt in track]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, s=1, c='blue')
        plt.gca().invert_yaxis()  # Match image coordinate system
        plt.title("Track Path (Black Pixels on Transparent PNG)")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


