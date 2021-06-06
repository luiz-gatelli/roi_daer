import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
from ast import literal_eval

_ids = ["2","3","4"]
models = ["vric","mars-small128"]

for _id in _ids:
    for model in models:
            filename = f"RUBEM_BERTA_{_id}_60_6_{model}"
            data = pd.read_csv(f"./data/{filename}.csv", sep=";", names=[
                            "track_id", "bbox", "class", "frame"], converters={"bbox": literal_eval})

            videoname = "RUBEM_BERTA_2"

            vidcap = cv2.VideoCapture(f"./data/{videoname}.mp4")
            WIDTH = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            HEIGHT = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            my_dpi = 96
            MIN_DISTANCE = 50

            data["x"] = (data.bbox.str[0] +
                        data.bbox.str[2]) / 2
            data["y"] = -(data.bbox.str[1] +
                        data.bbox.str[3]) / 2 + HEIGHT

            plt.figure(figsize=(WIDTH/my_dpi, HEIGHT/my_dpi), dpi=my_dpi)
            #img = plt.imread("./old_images/first_frame.png")
            segment = data[(data.frame < 5300) & (data.frame > 5000)]
            plt.scatter(segment.x, segment.y, c=segment.track_id, cmap='jet', s=12)
            plt.xlabel("X")
            plt.ylabel("Y")
            #plt.colorbar(label="Frame Index")
            plt.tight_layout()
            plt.savefig(f"./figures_10s/scatter_{filename}_full_nocb_10s.png", dpi=96)

