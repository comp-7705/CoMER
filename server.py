from typing_extensions import Annotated
import base64

import numpy as np
import cv2 as cv
import torch
from torchvision.transforms import ToTensor
from PIL import Image

from comer.lit_comer import LitCoMER
from comer.datamodule import vocab

from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()
device = "cpu"
ckp_path = "lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt"
model = LitCoMER.load_from_checkpoint(ckp_path, map_location=device)
model.eval()

def init_model(model=model):
    return model

class Buffer(BaseModel):
    payload: str
    
@app.post("/api/comer/prediction/")
async def predict(buffer: Buffer, model: Annotated[LitCoMER, Depends(init_model)]):
    img_buffer = base64.b64decode(buffer.payload)
    img_array = np.frombuffer(img_buffer, np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_ANYDEPTH)
    img = Image.fromarray(np.array(img))

    img = ToTensor()(img).to(device)
    mask = torch.zeros_like(img, dtype=torch.bool)
    hyp = model.approximate_joint_search(img.unsqueeze(0), mask)[0]
    pred_latex = vocab.indices2label(hyp.seq)

    return pred_latex
