from typing import List, Tuple
from pydantic import BaseModel
from ultralytics import YOLOWorld
from transformers.pipelines import pipeline
import torch, base64
from io import BytesIO
from google.cloud import aiplatform

# The pre-built serving docker image. It contains serving scripts and models.
SERVE_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-transformers-serve"
# from transformers import AutoProcessor, BlipModel
from PIL import Image

# @title Deploy with customized configs

# The pre-built serving docker image. It contains serving scripts and models.
SERVE_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-transformers-serve"
MODEL_ID = "Salesforce/blip-image-captioning-base"
accelerator_type = (
    "NVIDIA_L4"  # @param ["NVIDIA_L4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_T4"]
)
TASK = "image-to-text"
accelerator_count = 1
machine_type = "g2-standard-8"
use_dedicated_endpoint = True


def deploy_model(
    model_name: str,
    model_id: str,
    task: str,
    machine_type: str = "g2-standard-8",
    accelerator_type: str = "NVIDIA_L4",
    accelerator_count: int = 1,
    use_dedicated_endpoint: bool = True,
) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
    model_name = "blip-image-captioning"
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{model_name}-endpoint",
        dedicated_endpoint_enabled=use_dedicated_endpoint,
    )
    serving_env = {
        "MODEL_ID": model_id,
        "TASK": task,
        "DEPLOY_SOURCE": "notebook",
    }
    # If the model_id is a GCS path, use artifact_uri to pass it to serving docker.
    artifact_uri = model_id if model_id.startswith("gs://") else None
    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=SERVE_DOCKER_URI,
        serving_container_ports=[7080],
        serving_container_predict_route="/predictions/transformers_serving",
        serving_container_health_route="/ping",
        serving_container_environment_variables=serving_env,
        artifact_uri=artifact_uri,
        model_garden_source_model_name="publishers/salesforce/models/blip-image-captioning-base",
    )
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        deploy_request_timeout=1800,
        system_labels={
            "NOTEBOOK_NAME": "model_garden_pytorch_blip_image_captioning.ipynb"
        },
    )
    return model, endpoint


deplyed = True

# common_util.check_quota(
#     project_id=PROJECT_ID,
#     region=REGION,
#     accelerator_type=accelerator_type,
#     accelerator_count=accelerator_count,
#     is_for_training=False,
# )


yolo_model = YOLOWorld("yolov8s-world.pt")


class InferenceInput(BaseModel):
    id: int
    data: bytes
    link: str


class InferenceOutput(BaseModel):
    name: str
    imgbytestring: str
    description: str
    description_embedding: List[float] | None
    img_embedding: List[float] | None


class VertexBLIPInference:
    def __init__(self):

        # self.model, self.entrypoint = deploy_model(
        #     model_name="blip-image-captioning",
        #     model_id=MODEL_ID,
        #     task=TASK,
        #     machine_type=machine_type,
        #     accelerator_type=accelerator_type,
        #     accelerator_count=accelerator_count,
        #     use_dedicated_endpoint=use_dedicated_endpoint,
        # )

        self.endpoint = aiplatform.Endpoint(
            "projects/671807507414/locations/us-central1/endpoints/7683946906317225984"
        )

    def caption(self, src, caption: str = "Describe the image in detail."):
        if isinstance(src, str):
            image = Image.open(src).convert("RGB")
        elif isinstance(src, Image.Image):
            image = src.convert("RGB")
        else:
            raise ValueError("src must be a file path or PIL Image")

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        result = self.endpoint.predict(
            instances=[{"image": b64_image, "prompt": caption}]
        )

        return result
