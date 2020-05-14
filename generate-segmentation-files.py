#!/usr/bin/env bash
import json
import os
import numpy as np
import cv2


def generate_segmentation_file(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    for idx, v in enumerate(imgs_anns.values()):
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        # Because we only have one object category (balloon) to train,
        # 1 is the category of the background
        segmentation = np.ones((height, width), dtype=np.uint8)

        annos = v["regions"]
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = np.array(poly, np.int32)

            category_id = 0  # change to 255 for better visualisation
            cv2.fillPoly(segmentation, [poly], category_id)
            output = os.path.join(img_dir, "segmentation", v["filename"])
            cv2.imwrite(output, segmentation)


if "__main__" == __name__:
    for d in ["train", "val"]:
        os.makedirs(os.path.join("balloon", d, "segmentation"), exist_ok=True)
        generate_segmentation_file(os.path.join("balloon", d))
