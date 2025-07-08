from pydantic import BaseModel
from ultralytics import YOLOWorld
from transformers.pipelines import pipeline
import torch, base64

yolo_model = YOLOWorld("yolov8s-world.pt")


labeling_pipeline = pipeline(
    task="visual-question-answering",
    model="Salesforce/blip-vqa-base",
    torch_dtype=torch.float32,
    device=1,
)


class InferenceInput(BaseModel):
    id: int
    data: bytes
    link: str


class InferenceOutput(BaseModel):
    name: str
    imgbytestring: str
    description: str
