from fastapi import FastAPI
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pydantic import BaseModel

app = FastAPI()

class PlotData(BaseModel):
    x_data: list[float]
    y_data: list[float]
    title: str
    x_label: str = "Position (m)"
    y_label: str = "Deflection (m)"

@app.post("/plot")
async def generate_plot(data: PlotData):
    plt.figure(figsize=(10, 6))
    plt.plot(data.x_data, data.y_data)
    plt.title(data.title)
    plt.xlabel(data.x_label)
    plt.ylabel(data.y_label)
    plt.grid(True)
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    
    return {"plot": base64.b64encode(buf.read()).decode("utf-8")}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}