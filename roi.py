import json

import cv2 as cv2
import matplotlib
import matplotlib.figure as figure
import matplotlib.patches as patches
from ast import literal_eval

import pickle

import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.widgets import RectangleSelector
from PIL import Image

# The important line!
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Image general definitions
filename = "La_Grange_Kentucky_USA_60_6_mars-small128"
videoname = "KFC"


data = pd.read_csv(f"./data/{filename}.csv", sep=";", names=[
                   "track_id", "bbox", "class", "frame"], converters={"bbox": literal_eval})

vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
WIDTH = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
my_dpi = 96
MIN_DISTANCE = 100

print(WIDTH,HEIGHT)

fps = vidcap.get(cv2.CAP_PROP_FPS)

# Get the total numer of frames in the video.
frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

# Calculate the duration of the video in seconds
duration = 1

second = 0
vidcap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)  # optional
success, image = vidcap.read()

while success and second <= duration:
    # do stuff
    second += 1
    vidcap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    success, image = vidcap.read()

_, image = vidcap.read()
cv2.imwrite("first_frame.png", image)


data["x"] = (data.bbox.str[0] +
             data.bbox.str[2]) / 2
data["y"] = -(data.bbox.str[1] +
              data.bbox.str[3]) / 2 + HEIGHT

grouped = data.groupby("track_id")


def plot_arrow(data, color):
    plt.scatter(data["x"].values, data["y"].values,
                c='blue', marker='o', s=10, zorder=3)
    plt.scatter(data["last_x"].values,
                data["last_y"].values, c='red', marker='o', s=10, zorder=2)

    plt.quiver(data["x"], data["y"], ((data["last_x"] - data["x"])), ((data["last_y"] - data["y"])),
                                 angles="xy", scale_units="xy", scale=1, width=0.0015, color=color)
    #plt.quiver(data["x"].values, data["y"].values, ((x - data["x"]).values),((y - data["y"]).values), angles="xy", scale_units="xy", scale=1, width=0.002, color=color
    #for index, row in data.iterrows():
    #    for x, y in zip(row["last_3x"], row["last_3y"]):
    #       plt.quiver(row["x"], row["y"], ((x - row["x"])), ((y - row["y"])),
    #                  angles="xy", scale_units="xy", scale=1, width=0.0015, color=color)

data = pd.DataFrame()

last_3x = []
last_3y = []
last_bboxes = []
avg_distance = []
for _, group in grouped:
    last_3x.append(group.tail(3).x.values)
    last_3y.append(group.tail(3).y.values)
    last_bboxes.append(list(group.tail(3).bbox.values))
    data = data.append(group.head(1))


data["last_3x"] = last_3x
data["last_3y"] = last_3y
data["last_x"] = data["last_3x"].str[2]
data["last_y"] = data["last_3y"].str[2]
data["last_bboxes"] = last_bboxes

#for index, row in data.iterrows():
#    temp = 0
#    for x, y in zip(row["last_3x"], row["last_3y"]):
#        temp += (((x - row["x"])**2) + ((y - row["y"])**2))**0.5
#    avg_distance.append(temp/3)
#data["average_distance"] = avg_distance
data["distance"] = ((data.x - data.last_x)**2 +
                    ((data.y - data.last_y)**2))**0.5

## filtering data
data = data[data["distance"] > MIN_DISTANCE]


data["est_dx"] = (data.last_3x.str[2] - data.last_3x.str[0])/3
data["est_dy"] = (data.last_3y.str[2] - data.last_3y.str[0])/3


fig = plt.figure(figsize=(WIDTH/my_dpi, HEIGHT/my_dpi), dpi=my_dpi)
img = plt.imread("first_frame.png")
plt.imshow(img, extent=[0, WIDTH, 0, HEIGHT])

plot_arrow(data.loc[data['class'] == 'car'], 'black')
plot_arrow(data.loc[data['class'] == 'motorcycle'], 'green')
plot_arrow(data.loc[data['class'] == 'bus'], 'orange')
plot_arrow(data.loc[data['class'] == 'truck'], 'purple')

plt.axis('off')
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig(f"detected_arrows_{filename}.png", figsize=(
    WIDTH/my_dpi, HEIGHT/my_dpi), dpi=my_dpi)

plt.savefig(f"composed_image_{filename}.png",
            bbox_inches='tight', pad_inches=0, dpi=my_dpi, figsize=(WIDTH/my_dpi, HEIGHT/my_dpi))

im = cv2.imread(f"composed_image_{filename}.png")
rois = cv2.selectROIs("RoI Selector", im)

# roi = (x1,y1,w,h) (top left)

# cv2.destroyAllWindows()

rectangle_coords = [[roi[0], HEIGHT - roi[1] -
                     roi[3], roi[2], roi[3]] for roi in rois]

for index, rect in enumerate(rectangle_coords, start=1):
    print(f"Rectangle {index}: {rect}")

rectangles = [patches.Rectangle((coord[0], coord[1]), coord[2], coord[3],
                                color="lime", linewidth=2, fill=False) for coord in rectangle_coords]

rects = dict(enumerate(rectangles, start=1))

ax = plt.gca()

for r in rects:
    ax.add_artist(rects[r])
    rx, ry = rects[r].get_xy()
    cx = rx + rects[r].get_width()/2.0
    cy = ry + rects[r].get_height()/2.0

    ax.annotate(r, (cx, cy), color='lime', weight='bold',
                fontsize=72, ha='center', va='center')

plt.savefig(f"regions_{filename}.png",
            bbox_inches='tight', pad_inches=0)


def verifyIfInRect(x, y):
    for number, bbox in enumerate(rectangle_coords, start=1):
        # print(bbox[0] , (bbox[0] + bbox[2]), bbox[1] , (bbox[1] + bbox[3]))
        if bbox[0] <= x <= (bbox[0] + bbox[2]) and bbox[1] <= y <= (bbox[1] + bbox[3]):
            return number
    return 0


data["entry_box"] = data.apply(lambda row: verifyIfInRect(
    row["x"], row["y"]), axis=1)

data["exit_box"] = data.apply(lambda row: verifyIfInRect(
    row["last_3x"][-1], row["last_3y"][-1]), axis=1)



data.to_csv(f"classified_data_{filename}.csv")
pickle.dump(data,open(f"classified_data_{filename}.pkl","wb"))
