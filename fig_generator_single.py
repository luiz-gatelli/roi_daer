import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
from ast import literal_eval

#_id = 2 
# model = "vric"
_time = 0 # Initial time in frames
_delta = 300 # frame delta - duration of the logging in the image

#filename = f"RUBEM_BERTA_{_id}_60_6_{model}"
filename = "" ## name of the csv data file - that was provided by the system output
videoname = "KFC" ## use the videoname to get the picture sizes

data = pd.read_csv(f"./data/{filename}.csv", sep=";", names=[
                "track_id", "bbox", "class", "frame"], converters={"bbox": literal_eval}) ## load the CSV


vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4") # start the video capture
WIDTH = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)) # calculate the width
HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # calculate the height
my_dpi = 96 # dpi of the monitor - image quality

data["x"] = (data.bbox.str[0] +
            data.bbox.str[2]) / 2 # recalc the data x position for all entries
data["y"] = -(data.bbox.str[1] +
            data.bbox.str[3]) / 2 + HEIGHT # recalc the data y position for all entries

plt.figure(figsize=(WIDTH/my_dpi, HEIGHT/my_dpi), dpi=my_dpi) # start a figure plot
#img = plt.imread("./old_images/first_frame.png")
segment = data[(data.frame < _time + _delta) & (data.frame > _time)] # segment of data that we are interested in
#segment = data
plt.scatter(segment.x, segment.y, c=segment.track_id, cmap='jet', s=12) # plot the scatter with the colormap being the frame index
plt.xlabel("X")
plt.ylabel("Y")
#plt.colorbar(label="Frame Index")
plt.tight_layout() # tight layout
plt.savefig(f"./rb2/scatter_{filename}_{_time}_{_delta}.png", dpi=96) # save the figure

