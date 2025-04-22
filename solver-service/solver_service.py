from fastapi import FastAPI
import uvicorn
import numpy as np
from scipy.linalg import solve
from scipy.integrate import quad
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import math
from pydantic import BaseModel
from typing import List

app = FastAPI()

class SolverRequest(BaseModel):
    k: float
    N: int
    method: str  # 'finite_difference' или 'spline'

class SolverResponse(BaseModel):
    x_nodes: List[float]
    deflection: List[float]
    first_deriv: List[float]
    second_deriv: List[float]
    third_deriv: List[float]
    max_error: float

def exact_solution(x: float, k: float, A: float, EI: float, l: float) -> float:
    """Аналитическое решение уравнения изгиба балки"""
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

def finite_difference_solution(k: float, N: int, l: float, EI: float, A: float) -> dict:
    """Решение методом конечных разностей"""
    h = l / (N - 1)
    x_nodes = np.linspace(0, l, N)
    
    # Строим матрицу для 4-й производной
    main_diag = 6 * np.ones(N-4)
    off_diag = -4 * np.ones(N-5)
    diagonals = [np.ones(N-6), off_diag, main_diag, off_diag, np.ones(N-6)]
    W = diags(diagonals, [-3, -2, -1, 0, 1, 2, 3], shape=(N-4, N)).toarray() / h**4
    
    # Добавляем граничные условия
    # y(0) = 0
    W = np.vstack([np.eye(N)[0], W])
    # y'(0) = 0
    W = np.vstack([np.array([-3, 4, -1] + [0]*(N-3))/(2*h), W])
    # y(l) = 0
    W = np.vstack([W, np.eye(N)[-1]])
    # y'(l) = 0
    W = np.vstack([W, (np.array([0]*(N-3) + [1, -4, 3]))/(2*h)])
    
    # Правая часть
    F = np.array([A*x**k/EI for x in x_nodes])
    F[0] = 0  # y(0) = 0
    F[1] = 0  # y'(0) = 0
    F[-1] = 0  # y(l) = 0
    F[-2] = 0  # y'(l) = 0
    
    # Решение системы
    y = solve(W, F)
    
    # Вычисление производных
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

def spline_solution(k: float, N: int, l: float, EI: float, A: float) -> dict:
    """Решение с использованием сплайн-аппроксимации"""
    h = l / (N - 1)
    x_nodes = np.linspace(0, l, N)
    
    # Инициализация матрицы жесткости
    K = np.zeros((N, N))
    F = np.zeros(N)
    
    # Заполнение матрицы и вектора нагрузки
    for i in range(1, N-1):
        K[i, i-1] = 1/h**3
        K[i, i] = -2/h**3
        K[i, i+1] = 1/h**3
        
        # Интегрирование нагрузки
        x_prev = x_nodes[i-1]
        x_next = x_nodes[i+1]
        F[i] = quad(lambda x: A*x**k * (x - x_nodes[i-1])/h, x_prev, x_next)[0]
        F[i] += quad(lambda x: A*x**k * (x_nodes[i+1] - x)/h, x_prev, x_next)[0]
    
    # Граничные условия
    K[0, 0] = 1  # y(0) = 0
    K[-1, -1] = 1  # y(l) = 0
    
    # Решение системы
    y = solve(K, F/EI)
    
    # Вычисление производных через сплайны
    y_prime = np.zeros(N)
    y_double_prime = np.zeros(N)
    y_triple_prime = np.zeros(N)
    
    for i in range(1, N-1):
        y_prime[i] = (y[i+1] - y[i-1])/(2*h)
        y_double_prime[i] = (y[i+1] - 2*y[i] + y[i-1])/h**2
        y_triple_prime[i] = (y[i+2] - 2*y[i+1] + 2*y[i-1] - y[i-2])/(2*h**3)
    
    return {
        'x_nodes': x_nodes.tolist(),
        'deflection': y.tolist(),
        'first_deriv': y_prime.tolist(),
        'second_deriv': y_double_prime.tolist(),
        'third_deriv': y_triple_prime.tolist()
    }

def calculate_errors(numerical_sol: dict, k: float, l: float, EI: float, A: float) -> float:
    """Вычисление максимальной ошибки"""
    errors = []
    for x, y_num in zip(numerical_sol['x_nodes'], numerical_sol['deflection']):
        y_exact = exact_solution(x, k, A, EI, l)
        errors.append(abs(y_num - y_exact))
    return max(errors)

@app.post("/calculate", response_model=SolverResponse)
async def calculate(request: SolverRequest):
    l_beam = 2.0
    EI = 2e11 * 2 * 1e-8
    A = 6e2 / (l_beam**request.k)
    
    if request.method == 'finite_difference':
        solution = finite_difference_solution(request.k, request.N, l_beam, EI, A)
    elif request.method == 'spline':
        solution = spline_solution(request.k, request.N, l_beam, EI, A)
    else:
        raise ValueError("Неизвестный метод решения")
    
    max_error = calculate_errors(solution, request.k, l_beam, EI, A)
    
    return {
        "x_nodes": solution['x_nodes'],
        "deflection": solution['deflection'],
        "first_deriv": solution['first_deriv'],
        "second_deriv": solution['second_deriv'],
        "third_deriv": solution['third_deriv'],
        "max_error": max_error
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # Запуск на портах, указанных в docker-compose.yml 
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Для solver-service