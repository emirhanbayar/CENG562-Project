import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os

# === Configuration ===
# Update these paths accordingly
ground_truth_path = '/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017_original.json'  # Path to COCO ground truth annotations
predictions_path = '/home/emirhan/deteval/predictions_DINO.json'         # Path to your predictions.json file
img_dir = "deneme"
# =====================
# =====================

def main():
    # Initialize COCO ground truth
    print("Loading ground truth annotations...")
    coco_gt = COCO(ground_truth_path)

    # Load predictions
    print("Loading predictions...")
    with open(predictions_path, 'r') as f:
        predictions_data = json.load(f)
        predictions = predictions_data.get('predictions', [])

    if not predictions:
        print("No predictions found in the predictions.json file.")
        return

    # Convert predictions to COCO format
    print("Converting predictions to COCO format...")
    coco_dt = coco_gt.loadRes(predictions)

    #select one image
    imgIds = [139]
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox', img_dir=img_dir)
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    #now I will use the dict produced by the evaluate() function to calculate all

    #here i simply select the dict that corresponds to 'aRng': [0, 10000000000.0]
    image_evaluation_dict = coco_eval.evalImgs[0]

    #select the index related to IoU = 0.5
    iou_treshold_index = 0

    #all the detections from the model, it is a numpy of True/False (In my case they are all False)
    detection_ignore = image_evaluation_dict["dtIgnore"][iou_treshold_index]

    #here we consider the detection that we can not ignore (we use the not operator on every element of the array)
    mask = ~detection_ignore

    #detections number
    n_ignored = detection_ignore.sum()

    #and finally we calculate tp, fp and the total positives
    tp = (image_evaluation_dict["dtMatches"][iou_treshold_index][mask] > 0).sum()
    fp = (image_evaluation_dict["dtMatches"][iou_treshold_index][mask] == 0).sum()
    n_gt = len(image_evaluation_dict["gtIds"]) - image_evaluation_dict["gtIgnore"].astype(int).sum()

    recall = tp / n_gt
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")

if __name__ == '__main__':
    main()