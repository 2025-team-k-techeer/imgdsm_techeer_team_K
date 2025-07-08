from fastapi import FastAPI, Body
import uvicorn
from typing import Annotated, List
from cropper import Inference
from inference import InferenceInput, InferenceOutput

app = FastAPI()


@app.get("/")
def home():
    return {"api functional": "yes"}


@app.post("/segment")
def segment(
    new_request: Annotated[InferenceInput, Body()], response_model=List[InferenceOutput]
):
    inference: Inference
    if new_request.data:
        new_data = new_request.data
        inference = Inference(new_data)
    elif new_request.link:
        link = new_request.link
        inference = Inference(link)
    return inference.run_pipeline()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
