from fastapi import FastAPI
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from pydantic import BaseModel

app = FastAPI()

class PlotRequest(BaseModel):
    x_data: list[float]
    y_data: list[float]
    title: str

@app.post("/generate_plot")
async def generate_plot(request: PlotRequest):
    plt.figure()
    plt.plot(request.x_data, request.y_data)
    plt.title(request.title)
    plt.xlabel("Position (m)")
    plt.ylabel("Deflection (m)")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    return {"plot": base64.b64encode(buf.read()).decode("utf-8")}
@app.get("/health")
async def health():
    return {"status": "ok"}