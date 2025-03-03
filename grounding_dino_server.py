import asyncio
import io
import os
import sys

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from torchvision.ops import box_convert

from GroundingDINO.groundingdino.util.inference import load_model, predict
from server_utils import load_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration and model setup
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
FP16_INFERENCE = True

# Initialize FastAPI
app = FastAPI()

lock = asyncio.Lock()


# Load model during startup
@app.on_event("startup")
def load_groundingdino_model():
    global model
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    if FP16_INFERENCE and DEVICE == "cuda":
        model = model.half()


@app.post("/predict/")
async def predict_endpoint(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        async with lock:
            form = await request.form()

            # Read image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            image = load_image(image)

            # Preprocess image
            # image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            image_tensor = image[1]
            if FP16_INFERENCE and DEVICE == "cuda":
                image_tensor = image_tensor.half()

            # Perform prediction
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=form["text_prompt"],
                box_threshold=float(form["box_threshold"]),
                text_threshold=float(form["text_threshold"]),
                device=DEVICE,
            )

            boxes = boxes * torch.Tensor([image[0].width, image[0].height, image[0].width, image[0].height])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

            boxes_output = []
            labels = []
            confidences = []
            for box in xyxy:
                boxes_output.append([box[0].item(), box[1].item(), box[2].item(), box[3].item()])

            for phrase in phrases:
                labels.append(phrase)

            for logit in logits:
                confidences.append(logit.item())

            return {
                "boxes": boxes_output,
                "labels": labels,
                "logits": confidences,
            }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
