from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
from pydantic import BaseModel

app = FastAPI()
SOLVER_SERVICE = "http://0.0.0.0:8002"
VISUALIZATION_SERVICE = "http://0.0.0.0:8003"

class CalculationRequest(BaseModel):
    k: float
    N: int

@app.post("/full-analysis")
async def full_analysis(request: CalculationRequest):
    async with httpx.AsyncClient() as client:
        # Вызываем solver service
        solver_response = await client.post(
            f"{SOLVER_SERVICE}/calculate",
            json={"k": request.k, "N": request.N}
        )
        
        if solver_response.status_code != 200:
            raise HTTPException(status_code=502, detail="Solver service error")
        
        data = solver_response.json()
        
        # Вызываем visualization service
        plot_response = await client.post(
            f"{VISUALIZATION_SERVICE}/generate_plot",
            json={
                "x_data": data["x_nodes"],
                "y_data": data["deflection"],
                "title": f"Beam Deflection (k={request.k}, N={request.N})"
            }
        )
        
        if plot_response.status_code != 200:
            raise HTTPException(status_code=502, detail="Visualization service error")
        
        return {
            "calculation": data,
            "plot": plot_response.json()["plot"]
        }
    
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Порт из docker-compose.yml