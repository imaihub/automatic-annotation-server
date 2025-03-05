from typing import Optional

import cv2
import io as b_io

import numpy as np
import requests
from requests import Response

from .bbox import BoundingBox


def parse_grounding_dino_server_response(response: Response) -> list[BoundingBox]:
    """
    Parse the response from the predict endpoint in a Grounding Dino Server

    :param response: request.Response object returned by request.post
    :return: a list of BoundingBox objects
    """
    json_response = response.json()
    response_boxes = []
    for box, label, confidence in zip(json_response["boxes"], json_response["labels"], json_response["logits"]):
        box_object = BoundingBox(class_name=label)
        box_object.set_minmax_xy(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
        box_object.confidence = confidence
        response_boxes.append(box_object)

    return response_boxes


def predict_grounding_dino_external(image_np: np.ndarray, text_prompt: str, box_threshold: float, text_threshold: float, ip_address: Optional[str] = "localhost") -> list[BoundingBox]:
    """
    This function is used to send an inference request to a Grounding Server HTTP POST endpoint /predict, this response gets parsed into a list of bounding boxes in parse_predictions

    :param ip_address: IP address hosting the inference server
    :param image_np: numpy array with shape (H, W, 3) and assuming it has been converted to RGB already if loaded in through cv2.imread
    :param text_prompt: tells Grounding Dino which classes to detect
    :param text_threshold: text confidence threshold
    :param box_threshold: box confidence threshold

    :return: list of prediction bounding boxes
    """
    _, buffer = cv2.imencode(".png", image_np)
    image_buffer = b_io.BytesIO(buffer)
    image_buffer.seek(0)

    files = {"file": image_buffer.read()}

    data = {
        "text_prompt": text_prompt,
        "text_threshold": text_threshold,
        "box_threshold": box_threshold,
    }

    content = requests.post(f"http://{ip_address}:8901/predict/", data=data, files=files)
    prediction_boxes = parse_grounding_dino_server_response(content)
    return prediction_boxes
