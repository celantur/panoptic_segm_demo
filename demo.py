import json
import os
import random

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg


# The following is modification of Detectron2 Beginner's Tutorial.
# Cf https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        # Pixel-wise segmentation
        record["sem_seg_file_name"] = os.path.join(img_dir, "segmentation", v["filename"])

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                # "Things" are well-defined countable objects,
                # while "stuff" is amorphous something with a different label than the background.
                "isthing": True,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def visualise_dataset(d: str = "train"):
    dataset_dicts = get_balloon_dicts(os.path.join("balloon", d))
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualise_prediction(predictor, d: str = "val"):
    dataset_dicts = get_balloon_dicts(os.path.join("balloon", d))
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow(d["file_name"], v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def init_cfg(config_file: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    return cfg


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def get_predictor(cfg, model_name: str):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)
    return predictor


if "__main__" == __name__:
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        # For semantic / panoptic segmentation, add a stuff class.
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], stuff_classes=[])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    # Check whether dataset is correctly initialised
    visualise_dataset("train")

    # Setup configuration
    cfg = init_cfg("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    #cfg = init_cfg("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Vanilla single-GPU training
    # train(cfg)

    # Multi-GPU training
    launch(
        train,
        1,  # Number of GPUs per machine
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:1234",
        args=(cfg,),
    )


    # Check result
    predictor = get_predictor(cfg, "model_final.pth")
    visualise_prediction(predictor, "val")

