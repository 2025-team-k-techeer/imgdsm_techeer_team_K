from io import BytesIO
from typing import List
from inference import InferenceOutput, yolo_model, labeling_pipeline
from PIL import Image
import base64
import numpy as np
import cv2


class Inference:
    def __init__(self, input_data):
        self.data = self.data = base64.b64decode(input_data)
        print(type(self.data))
        self.image = np.frombuffer(self.data, dtype=np.uint8)
        self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError("Failed to decode image: data may not be a valid image.")
        # Prepare PIL Image

    def crop_picture(self):
        print(np.frombuffer(self.data, dtype=np.uint8))

        # Image.open(BytesIO(self.data)).convert("RGB")
        print("Calling YOLO...")
        results = yolo_model.predict(self.image, conf=0.10)
        print("YOLO done.")
        return results

    def label_picture(self, results) -> List[InferenceOutput]:
        labeled_pictures = []
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.image)
        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy())
            label_name = yolo_model.names[class_id]
            crop_box = box.xyxy[0].cpu().numpy()

            # Crop region
            cropped = self.image.crop(crop_box)

            # Label with your captioning model
            result = labeling_pipeline(
                question=f"Describe the {label_name}'s color, texture, and style, only adjectives.",
                image=cropped,
                generate_kwargs={"num_beams": 3, "do_sample": False},
            )

            if not result or not isinstance(result, list):
                continue

            description = result[0].get("answer", "unknown")

            # Convert cropped image to bytes
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            # Append to results
            labeled_pictures.append(
                InferenceOutput(
                    name=label_name,
                    imgbytestring=base64.b64encode(img_bytes).decode("utf-8"),
                    description=description,
                )
            )

        return labeled_pictures

    def run_pipeline(self):
        return self.label_picture(self.crop_picture())
