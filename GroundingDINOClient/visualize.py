import cv2
import numpy as np
from typing import Optional
from GroundingDINOClient.bbox import BoundingBox


def draw_bounding_boxes(img: np.ndarray, bounding_boxes: list[BoundingBox], color: Optional[tuple] = (255, 0, 0), width: Optional[int] = 2) -> np.ndarray:
    """
    Draw bounding boxes on an image

    :param img: image to draw bounding boxes on
    :param bounding_boxes: BoundingBox objects to draw
    :param color: color of bounding boxes
    :param width: width of the boxes' outlines

    :return: image with bounding boxes drawn
    """
    for box in bounding_boxes:
        if not box.class_name:  # In the case of not containing a class_name, visualize with a different color
            draw_rect(img=img, y1=box.y1, x1=box.x1, y2=box.y2, x2=box.x2, text=f"{box.class_name} {round(box.confidence, 2)}", color=color, width=width, fontscale=max(1, width // 3), thickness=max(1, width // 3))
            continue
        draw_rect(img=img, y1=box.y1, x1=box.x1, y2=box.y2, x2=box.x2, text=f"{box.class_name} {round(box.confidence, 2)}", color=(0, 0, 255), width=width, fontscale=max(1, width // 3), thickness=max(1, width // 3))

    return img


def draw_rect(img, y1, x1, y2, x2, text=None, color=(255, 0, 0), width=2, fontscale=1, thickness=1):
    if not img.data.contiguous:
        print("Warning: the array is not contiguous, the rectangle will probably not be visible.")
    y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, width)

    if text is not None:
        text = str(text)
        text_size = cv2.getTextSize(f"{text}", cv2.FONT_HERSHEY_PLAIN, fontscale, thickness)[0]
        cv2.rectangle(img, (x1 + 1, y1 + 1), (x1 + text_size[0] + 4, y1 + text_size[1] + 5), 0, -1)
        cv2.putText(img, f"{text}", (x1 + 1, y1 + text_size[1] + 5), cv2.FONT_HERSHEY_PLAIN, fontscale, color, thickness)

