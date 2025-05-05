from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
from scipy.linalg import solve
from scipy.integrate import quad
from scipy.sparse import diags
from pydantic import BaseModel, Field, validator
from typing import List, Literal
import logging
import math

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Константы задачи
BEAM_LENGTH = 2.0  # Длина балки (м)
EI = 2e11 * 2 * 1e-8  # Жесткость (Па·м⁴)
LOAD_AMPLITUDE = 6e2  # Амплитуда нагрузки (Н/м)
matrf = np.zeros((8, 8, 4))  # 8 methods x 8 k values x 4 N values

def pogr(a, toch, n):
    """Calculate relative error"""
    pogr_d = np.abs(a - toch[:-1])
    max_pogr = np.max(pogr_d, axis=0)
    return max_pogr[n] / (toch[-1, n] + 1e-16)  # Avoid division by zero

class SolverRequest(BaseModel):
    """Модель запроса с валидацией"""
    k: float = Field(..., gt=0, description="Показатель степени нагрузки (k > 0)")
    N: int = Field(..., ge=3, le=1000, description="Количество узлов (3 ≤ N ≤ 1000)")
    method: Literal['finite_difference', 'spline'] = Field(...,
        description="Метод решения: finite_difference|spline")

    @validator('N')
    def validate_odd_number(cls, v):
        if v % 2 == 0:
            logger.warning(f"Рекомендуется нечетное N. Получено: {v}")
        return v

class SolverResponse(BaseModel):
    """Модель ответа"""
    x_nodes: List[float]
    deflection: List[float]
    first_deriv: List[float]
    second_deriv: List[float]
    third_deriv: List[float]
    max_error: float

def exact_solution(x: float, k, A: float) -> float:
    """Аналитическое решение уравнения изгиба балки"""
    k=int(k)
    l = BEAM_LENGTH
    if k == 0:
        return (A/(24*EI)) * (x**4 - 2*l*x**3 + l**3*x)
    elif k == 1:
        return (A/(120*EI)) * (x**5 - (5/2)*l*x**4 + (5/3)*l**4*x)
    elif k == 2:
        return (A/(360*EI)) * (x**6 - 3*l*x**5 + (5/2)*l**4*x**2)
    else:
        n = k + 4
        return (A/(EI*math.factorial(n))) * (
            x**n - (n/(n-2))*l*x**(n-1) + (n/(n-3))*l**(n-3)*x**3)

def finite_difference_solution(k: float, N: int) -> dict:
    """Решение методом конечных разностей"""
    h = BEAM_LENGTH / (N - 1)
    x_nodes = np.linspace(0, BEAM_LENGTH, N)
    A = LOAD_AMPLITUDE / (BEAM_LENGTH**k)
    
    # Матрица для 4-й производной (центральные разности)
    # main_diag = 6 * np.ones(N-2)/h**2
    # off_diag = -4 * np.ones(N-3)/h**4
    # far_diag = np.ones(N - 4) / h**4
    # diagonals = [
    #     far_diag,      # offset -2
    #     off_diag,       # offset -1
    #     main_diag,      # offset 0 
    #     off_diag,       # offset +1
    #     far_diag        # offset +2
    # ]
    diagonals = [np.ones(N-4), -4*np.ones(N-3), 6*np.ones(N-2),
                        -4*np.ones(N-3), np.ones(N-4)]
    W = diags(
        diagonals,
        offsets=[-2, -1, 0, 1, 2],
        shape=(N - 2, N-2 )  # Важно: (N-2)x(N-2) вместо (N-4)xN
    ).toarray() / h**4
    
    # Граничные условия
    # W = np.vstack([
    #     np.eye(N)[0],  # y(0) = 0
    #     np.array([-3, 4, -1] + [0]*(N-3)) / (2*h),  # y'(0) = 0
    #     W,
    #     (np.array([0]*(N-3) + [1, -4, 3]))/(2*h),  # y'(l) = 0
    #     np.eye(N)[-1]  # y(l) = 0
    # ])
    
    # # Правая часть
    # F = np.array([A*x**k/EI for x in x_nodes])
    # F[0] = F[1] = F[-1] = F[-2] = 0  # Обнуляем граничные условия
    
    # Add boundary conditions
    W_full = np.zeros((N, N))
    W_full[1:-1, 1:-1] = W
    
    # Boundary conditions
    W_full[0, 0] = 1  # y(0) = 0
    W_full[-1, -1] = 1  # y(l) = 0
    W_full[1, :3] = np.array([-3, 4, -1])/(2*h)  # y'(0) = 0
    W_full[-2, -3:] = np.array([1, -4, 3])/(2*h)  # y'(l) = 0
    # Right-hand side
    F = np.zeros(N)
    F[1:-1] = np.array([A*x**k/EI for x in x_nodes[1:-1]])

    # Решение системы
    y = solve(W_full, F)
    
    # # Численные производные
    # y_prime = np.gradient(y, h, edge_order=2)
    # y_double_prime = np.gradient(y_prime, h, edge_order=2)
    # y_triple_prime = np.gradient(y_double_prime, h, edge_order=2)
    
    # Calculate derivatives using finite differences
    MKRF = np.zeros((N, 4))
    MKRF[:, 0] = y
    
    # First derivative
    MKRF[1:-1, 1] = (y[2:] - y[:-2])/(2*h)
    MKRF[0, 1] = 0  # Boundary condition
    MKRF[-1, 1] = 0  # Boundary condition
    
    # Second derivative
    MKRF[1:-1, 2] = (y[2:] - 2*y[1:-1] + y[:-2])/h**2
    MKRF[0, 2] = MKRF[1, 2]  # Approximate
    MKRF[-1, 2] = MKRF[-2, 2]  # Approximate
    
    # Third derivative
    MKRF[2:-2, 3] = (y[4:] - 2*y[3:-1] + 2*y[1:-3] - y[:-4])/(2*h**3)
    MKRF[:2, 3] = MKRF[2, 3]  # Approximate
    MKRF[-2:, 3] = MKRF[-3, 3]  # Approximate
    
    # ==============================================
    # Exact Solution Calculation
    # ==============================================
    Toch = np.zeros((N+1, 4))
    for i, x in enumerate(x_nodes):
        Toch[i, 0] = exact_solution(x, k, A)
        
        # Numerical derivatives
        h_small = 1e-5
        Toch[i, 1] = (exact_solution(x+h_small, k, A) - 
                     exact_solution(x-h_small, k, A))/(2*h_small)
        Toch[i, 2] = (exact_solution(x+h_small, k, A) - 
                     2*exact_solution(x, k, A) + 
                     exact_solution(x-h_small, k, A))/h_small**2
        Toch[i, 3] = (exact_solution(x+2*h_small, k, A) - 
                    2*exact_solution(x+h_small, k, A) + 
                    2*exact_solution(x-h_small, k, A) - 
                    exact_solution(x-2*h_small, k, A))/(2*h_small**3)
    
    # Max absolute values for normalization
    Toch[-1, :] = np.max(np.abs(Toch[:-1, :]), axis=0)
    
    # Store errors (just using FD method for demonstration)
    for i in range(4):
        matrf[i+4, 0, 0] = pogr(MKRF[:, :4], Toch, i)
        
    # except Exception as e:
    #     print(f"Error for k={k}, N={N}: {str(e)}")
    #     continue

    return {
        'x_nodes': x_nodes.tolist(),
        'deflection': y.tolist(),
        'first_deriv': Toch[:,1].tolist(),
        'second_deriv': Toch[:,2].tolist(),
        'third_deriv': Toch[:,3].tolist()
    }

def spline_solution(k: float, N: int) -> dict:
    """Решение с использованием сплайн-аппроксимации"""
    h = BEAM_LENGTH / (N - 1)
    x_nodes = np.linspace(0, BEAM_LENGTH, N)
    A = LOAD_AMPLITUDE / (BEAM_LENGTH**k)
    
    # Матрица жесткости
    K = np.zeros((N, N))
    F = np.zeros(N)
    
    for i in range(1, N-1):
        K[i, i-1] = 1/h**3
        K[i, i] = -2/h**3
        K[i, i+1] = 1/h**3
        
        # Интегрирование нагрузки
        def load_func(x):
            return A*x**k * ((x_nodes[i+1] - x)/h if i < N-1 else 0) + \
                   A*x**k * ((x - x_nodes[i-1])/h if i > 0 else 0)
        
        F[i] = quad(load_func, max(0, x_nodes[i-1]), min(BEAM_LENGTH, x_nodes[i+1]))[0]
    
    # Граничные условия
    K[0, 0] = K[-1, -1] = 1
    F[0] = F[-1] = 0
    
    # Решение
    y = solve(K, F/EI)
    
    # Производные через конечные разности
    y_prime = np.gradient(y, h, edge_order=2)
    y_double_prime = np.gradient(y_prime, h, edge_order=2)
    y_triple_prime = np.gradient(y_double_prime, h, edge_order=2)
    
    return {
        'x_nodes': x_nodes.tolist(),
        'deflection': y.tolist(),
        'first_deriv': y_prime.tolist(),
        'second_deriv': y_double_prime.tolist(),
        'third_deriv': y_triple_prime.tolist()
    }

def calculate_max_error(numerical_sol: dict, k: float) -> float:
    """Вычисление максимальной ошибки относительно аналитического решения"""
    A = LOAD_AMPLITUDE / (BEAM_LENGTH**k)
    errors = [
        abs(y_num - exact_solution(x, k, A))
        for x, y_num in zip(numerical_sol['x_nodes'], numerical_sol['deflection'])
    ]
    return max(errors)

@app.post("/calculate", response_model=SolverResponse)
async def calculate(request: SolverRequest):
    """Основной endpoint для расчетов"""
    logger.info(f"Получен запрос: {request.dict()}")
    
    try:
        if request.method == 'finite_difference':
            solution = finite_difference_solution(request.k, request.N)
        else:
            solution = spline_solution(request.k, request.N)
        
        max_error = calculate_max_error(solution, request.k)
        
        return {
            "x_nodes": solution['x_nodes'],
            "deflection": solution['deflection'],
            "first_deriv": solution['first_deriv'],
            "second_deriv": solution['second_deriv'],
            "third_deriv": solution['third_deriv'],
            "max_error": max_error
        }
        
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "ok", "timestamp": "2025-04-28T16:01:03"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
