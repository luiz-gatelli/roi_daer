
import numpy as np
from matplotlib.widgets import RectangleSelector
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2 as cv2
import json
import pandas as pd
import sklearn as sk
from PIL import Image

# Image general definitions
filename = "tracked_data_rubem_berta_90"
videoname = "RUBEM_BERTA_2"

vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
WIDTH = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
HEIGHT = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(WIDTH,HEIGHT)

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

# Load First Frame as Image
vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
_, image = vidcap.read()
cv2.imwrite("first_frame.png", image)

my_dpi = 96
# Matplotlib Figure - Initialize
fig, ax = plt.subplots(1, figsize=(WIDTH/my_dpi, HEIGHT/my_dpi), dpi=my_dpi)


def plot_arrow(ax, data, color):
    plt.scatter(data["x_initial"].values, data["y_initial"].values,
               c='blue', marker='o', s=10, zorder=3)
    plt.scatter(data["x_final"].values,
               data["y_final"].values, c='red', marker='o', s=10, zorder=2)

    plt.quiver(data["x_initial"].values, data["y_initial"].values, ((data["x_final"] - data["x_initial"]).values),
              ((data["y_final"] - data["y_initial"]).values), angles="xy", scale_units="xy", scale=1, width=0.002, color=color)



#ax.scatter(data["x_initial"].values, data["y_initial"].values,
#           c='blue', marker='o', s=10, zorder=3)
#ax.scatter(data["x_final"].values,
#           data["y_final"].values, c='red', marker='o', s=10, zorder=2)


plot_arrow(ax, data.loc[data['class'] == 'car'], 'black')

plot_arrow(ax, data.loc[data['class'] == 'motorcycle'], 'green')

plot_arrow(ax, data.loc[data['class'] == 'bus'], 'orange')

plot_arrow(ax, data.loc[data['class'] == 'truck'], 'purple')


ax.axis('off')

# plt.show()

fig.savefig(f"detected_arrows_{filename}.png",
            bbox_inches='tight', pad_inches=0)

img = plt.imread("first_frame.png")
plt.imshow(img, extent=[0, WIDTH, 0, HEIGHT])

fig.savefig(f"composed_image_{filename}.png",
            bbox_inches='tight', pad_inches=0, dpi=my_dpi, figsize=(1280/my_dpi, 720/my_dpi))

# resize image for match:
size = int(1280), int(720)

im = Image.open(f"composed_image_{filename}.png")
im.resize(size, Image.ANTIALIAS)
im.save(f"resized_image_{filename}.png")

# select RoIs with OpenCV

im = cv2.imread(f"resized_image_{filename}.png")

#im = cv2.imread(f"first_frame.png")
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

for r in rects:
    ax.add_artist(rects[r])
    rx, ry = rects[r].get_xy()
    cx = rx + rects[r].get_width()/2.0
    cy = ry + rects[r].get_height()/2.0

    ax.annotate(r, (cx, cy), color='lime', weight='bold',
                fontsize=15, ha='center', va='center')

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

data["exit_box"].describe()
data["entry_box"].describe()
