from io import BytesIO
from typing import List
from inference import InferenceOutput, yolo_model, VertexBLIPInference
from vertexai.vision_models import Image as gImage, MultiModalEmbeddingModel
from PIL import Image
import base64
import numpy as np
import cv2
import requests


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
        results = yolo_model.predict(self.image, conf=0.10)
        return results

    def label_picture(self, results, url="") -> List[InferenceOutput]:
        labeled_pictures = []
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(self.image)
        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy())
            label_name = yolo_model.names[class_id]
            crop_box = box.xyxy[0].cpu().numpy()

            # Crop region
            cropped = self.image.crop(crop_box)
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")

            img_bytes = buffer.getvalue()

            # Label with your captioning model

            vertexapi = VertexBLIPInference()
            result = vertexapi.caption(
                src=cropped, caption=f"describe {label_name} in the image."
            )
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            embedding_dimension = 128

            embeddings = model.get_embeddings(
                image=gImage(buffer.getvalue()),
                contextual_text="Colosseum",
                dimension=embedding_dimension,
            )

            if not result or not isinstance(result, list):
                continue

            # Convert cropped image to bytes

            # Append to results
            labeled_pictures.append(
                InferenceOutput(
                    name=label_name,
                    imgbytestring=base64.b64encode(img_bytes).decode("utf-8"),
                    description=result[0],
                    description_embedding=embeddings.text_embedding,
                    img_embedding=embeddings.image_embedding,
                )
            )

        return labeled_pictures

    # def get_text_embedding(self, caption):
    #     inputs = processor(text=[caption], return_tensors="pt", padding=True)
    #     return (
    #         model.get_text_features(
    #             input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    #         )
    #         .detach()
    #         .numpy()
    #         .tolist()
    #     )

    # def get_img_embedding(self, img):
    #     inputs = processor(images=img, return_tensors="pt", padding=True)
    #     return model.get_image_features(**inputs).detach().numpy().tolist()

    def run_pipeline(self):
        return self.label_picture(self.crop_picture())
