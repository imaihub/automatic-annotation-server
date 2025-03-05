import unittest
import os
import cv2
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate,
)

class TestGroundingDINO(unittest.TestCase):
    CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_PATH = "GroundingDINO/.asset/cat_dog.jpeg"
    TEXT_PROMPT = "Cat. Dog."
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    FP16_INFERENCE = True

    @classmethod
    def setUpClass(cls):
        """Load model and image for testing"""
        cls.image_source, cls.image = load_image(cls.IMAGE_PATH)
        cls.model = load_model(cls.CONFIG_PATH, cls.CHECKPOINT_PATH)

        if cls.FP16_INFERENCE and cls.DEVICE == "cuda":
            cls.image = cls.image.half()
            cls.model = cls.model.half()

    def test_model_load(self):
        """Test that the model loads successfully"""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'forward'))

    def test_image_load(self):
        """Test that the image loads successfully"""
        self.assertIsNotNone(self.image_source)
        self.assertEqual(len(self.image_source.size), 2)  # Check that it's an image-like array

    def test_prediction(self):
        """Test that the model performs predictions"""
        boxes, logits, phrases = predict(
            model=self.model,
            image=self.image,
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            device=self.DEVICE,
        )

        # Check that predictions were made
        self.assertGreater(len(boxes), 0, "No boxes were predicted")
        self.assertGreater(len(logits), 0, "No logits were returned")
        self.assertGreater(len(phrases), 0, "No phrases were returned")

    def test_annotation(self):
        """Test that the annotated frame is created without errors"""
        boxes, logits, phrases = predict(
            model=self.model,
            image=self.image,
            caption=self.TEXT_PROMPT,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            device=self.DEVICE,
        )

        annotated_frame = annotate(image_source=self.image_source, boxes=boxes, logits=logits, phrases=phrases)

        # Ensure that the annotated frame is generated
        self.assertIsNotNone(annotated_frame)
        self.assertEqual(annotated_frame.shape[2], 3)  # Check that it's a color image

        # Write the image to verify it can be saved successfully
        output_path = "test_annotated_image.jpg"
        cv2.imwrite(output_path, annotated_frame)
        self.assertTrue(os.path.exists(output_path), "Annotated image was not saved")

        # Cleanup the test file after writing
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
