"""
Known issues:

1. If a box does not contain a class_name, the text-threshold could be too high. In the visualization it is shown as a red box

"""

import argparse
import os
from glob import glob

import cv2
import numpy as np

from GroundingDINOClient.bbox import BoundingBox
from GroundingDINOClient.utils import predict_grounding_dino_external
from GroundingDINOClient.logger import Logger
from GroundingDINOClient.visualize import draw_bounding_boxes

parser = argparse.ArgumentParser()

parser.add_argument("--input-folder", type=str, default="assets/", help="Folder path to read out recursively")
parser.add_argument("--output-folder", type=str, default="output/", help="Folder to save annotations and visualization in")
parser.add_argument("--text-prompt", type=str, default="leaf. plant", help="Tells Grounding Dino which classes to detect, put fullstops between classes like the example shows")
parser.add_argument("--text-threshold", type=float, default=0.1, help="Text threshold")
parser.add_argument("--box-threshold", type=float, default=0.1, help="Confidence threshold for filtering out the boxes")
parser.add_argument("--min-area", type=float, default=1000, help="Min area for filtering out the boxes")
parser.add_argument("--max-area", type=float, default=100000, help="Max area for filtering out the boxes")
parser.add_argument("--extensions", type=str, default=".jpg,.png", help="Comma-separated list of accepted file extensions")
parser.add_argument("--annot-format", type=str, default="cvat", help="The annotations get saved into this format, currently the options cvat/yolo are supported. "
                                                                     "When chosen yolo, alongside the images from the input folder, annotations files get added")

args = parser.parse_args()
logger = Logger.setup_logger()

classes = [label.strip(",. ").lower() for label in args.text_prompt.split(".") if label]

os.makedirs(os.path.join(args.output_folder, "visualization"), exist_ok=True)

annotations_path = os.path.join(args.output_folder, 'annotations.xml')

def post_process(image_np: np.ndarray, prediction_boxes: list, file_name: str):
    image_with_boxes = draw_bounding_boxes(image_np, prediction_boxes, width=max(2, image_np.shape[1] // 500))
    cv2.imwrite(os.path.join(args.output_folder, "visualization", os.path.basename(file_name)), cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR))


def filter_missing_names(prediction_boxes: list[BoundingBox]) -> list[BoundingBox]:
    new_boxes = []
    for box in prediction_boxes:
        if box.class_name:
            new_boxes.append(box)
    return new_boxes

def filter_area_boxes(prediction_boxes: list[BoundingBox], min_area: int, max_area: int) -> list[BoundingBox]:
    new_boxes = []
    for box in prediction_boxes:
        if min_area < box.width * box.height < max_area:
            new_boxes.append(box)
    return new_boxes


def collect_image_files():
    files = []
    for extension in args.extensions.split(","):
        files.extend(glob(os.path.join(args.input_folder, "**", "*" + extension), recursive=True))

    return files


if __name__ == "__main__":
    image_files = collect_image_files()
    for i, file in enumerate(image_files):
        try:
            image = cv2.cvtColor(cv2.imread(file, -1), cv2.COLOR_BGR2RGB)
            predictions = predict_grounding_dino_external(image, text_prompt=args.text_prompt, text_threshold=args.text_threshold, box_threshold=args.box_threshold, ip_address="127.0.0.1") # 127.0.0.1 in case of running the server locally, 141.252.12.25 is not always available
            width = image.shape[1]
            height = image.shape[0]

            predictions = filter_area_boxes(prediction_boxes=predictions, min_area=args.min_area, max_area=args.max_area)

            post_process(image_np=image, prediction_boxes=predictions, file_name=file)

            if args.filter_missing_names:
                predictions = filter_missing_names(prediction_boxes=predictions)

            logger.info(f"Saving bboxes for {file}. {len(image_files) - i} images to go")
        except Exception as e:
            logger.exception(f"Could not process {file}: {e}")
            continue
