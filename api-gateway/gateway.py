from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
from enum import Enum
from typing import Literal
from pydantic import Field, BaseModel
from typing import Literal, Optional

app = FastAPI()

# Конфигурация сервисов
SOLVER_SERVICE_URL = "http://solver-service:8002"
VISUALIZATION_SERVICE_URL = "http://visualization-service:8003"

class MethodEnum(str, Enum):
    finite_difference = "finite_difference"
    spline = "spline"

class AdditionalParams(BaseModel):
    beam_length: float = Field(2.0, gt=0)
    load_amplitude: float = Field(600.0, gt=0)

class AnalysisRequest(BaseModel):
    k: float = Field(..., gt=0)
    N: int = Field(..., ge=3, le=1000)
    method: Literal['finite_difference', 'spline']
    additional_params: Optional[AdditionalParams] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/full-analysis")
async def full_analysis(request: AnalysisRequest):
    try:
        async with httpx.AsyncClient() as client:
            # 1. Запрос к solver-service
            solver_response = await client.post(
                f"{SOLVER_SERVICE_URL}/calculate",
                json={
                    "k": request.k,
                    "N": request.N,
                    "method": request.method
                },
                timeout=30.0
            )
            solver_response.raise_for_status()
            solver_data = solver_response.json()
            
            # 2. Генерация графика
            plot_title = f"Beam Deflection (k={request.k}, N={request.N})"
            if request.additional_params:
                plot_title += f" | L={request.additional_params.beam_length}"
            
            plot_response = await client.post(
                f"{VISUALIZATION_SERVICE_URL}/plot",
                json={
                    "x_data": solver_data["x_nodes"],
                    "y_data": solver_data["deflection"],
                    "title": plot_title
                },
                timeout=30.0
            )
            plot_response.raise_for_status()
            
            return {
                "calculation": solver_data,
                "plot": plot_response.json()
            }
            
    except httpx.ConnectError:
        raise HTTPException(503, detail="Backend service unavailable")
    except Exception as e:
        raise HTTPException(500, detail=str(e))
