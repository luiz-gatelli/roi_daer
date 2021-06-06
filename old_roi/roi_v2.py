

import json

import cv2 as cv2
import matplotlib
import matplotlib.figure as figure
import matplotlib.patches as patches

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
filename = "tracked_data_experimento_rubemberta_2_60"
videoname = "RUBEM_BERTA_2"



vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
WIDTH = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
MIN_DISTANCE = 50
print(f"Figure Dimensions: {WIDTH} x {HEIGHT}")

vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
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


# Load DataFrame
data = pd.read_json(f"./data/{filename}.json", 'records').T

data["x_initial"] = (data["initial position"].str[0] +
                     data["initial position"].str[2]) / 2
data["x_final"] = (data["last position"].str[0] +
                   data["last position"].str[2]) / 2
data["y_initial"] = -(data["initial position"].str[1] +
                      data["initial position"].str[3]) / 2 + HEIGHT
data["y_final"] = -(data["last position"].str[1] +
                    data["last position"].str[3]) / 2 + HEIGHT

data["distance"] = ((data.x_initial - data.x_final)**2 +
                    (data.y_initial - data.y_final)) ** 0.5

data = data[data["distance"] > MIN_DISTANCE]

# Load First Frame as Image
vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
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

my_dpi = 96
# Matplotlib Figure - Initialize
plt.figure(figsize=(WIDTH/my_dpi, HEIGHT/my_dpi), dpi=my_dpi)


def plot_arrow(data, color):
    plt.scatter(data["x_initial"].values, data["y_initial"].values,
                c='blue', marker='o', s=10, zorder=3)
    plt.scatter(data["x_final"].values,
                data["y_final"].values, c='red', marker='o', s=10, zorder=2)
    plt.quiver(data["x_initial"].values, data["y_initial"].values, ((data["x_final"] - data["x_initial"]).values),
               ((data["y_final"] - data["y_initial"]).values), angles="xy", scale_units="xy", scale=1, width=0.002, color=color)


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

img = plt.imread("first_frame.png")
plt.imshow(img, extent=[0, WIDTH, 0, HEIGHT])

plt.savefig(f"composed_image_{filename}.png",
            bbox_inches='tight', pad_inches=0, dpi=my_dpi, figsize=(1280/my_dpi, 720/my_dpi))

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
    row["x_initial"], row["y_initial"]), axis=1)

data["exit_box"] = data.apply(lambda row: verifyIfInRect(
    row["x_final"], row["y_final"]), axis=1)


data.to_csv(f"classified_data_{filename}.csv")
