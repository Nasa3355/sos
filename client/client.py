import httpx
import asyncio
import os
import sys
from typing import List, Literal

async def run_analysis():
    BASE_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Проверка здоровья API
            for _ in range(30):
                try:
                    resp = await client.get(f"{BASE_URL}/health")
                    if resp.status_code == 200:
                        break
                except httpx.RequestError:
                    await asyncio.sleep(1)
            else:
                raise Exception("API Gateway недоступен")

            response = await client.post(
                f"{BASE_URL}/full-analysis",
                json={
                    "k": 1.0,
                    "N": 21,
                    "method": "finite_difference",  # Обязательное поле
                    "additional_params": {          # Опционально
                        "beam_length": 2.0,
                        "load_amplitude": 600
                    }
                }
                #timeout=10.0
            )
            response.raise_for_status()
            print("Успешный ответ:", response.json())
            
    except httpx.HTTPStatusError as e:
        print(f"Ошибка сервера ({e.response.status_code}):", e.response.text)
        sys.exit(1)
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_analysis())
# import httpx
# import asyncio
# import os
# import sys
# import base64
# from io import BytesIO
# import matplotlib.pyplot as plt
# from PIL import Image

# async def run_analysis():
#     BASE_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")
    
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             # Проверка доступности сервиса
#             health = await client.get(f"{BASE_URL}/health")
#             health.raise_for_status()

#             # Отправка запроса на анализ
#             response = await client.post(
#                 f"{BASE_URL}/full-analysis",
#                 json={
#                     "k": 1.0,
#                     "N": 21,
#                     "method": "finite_difference"
#                 }
#             )
#             response.raise_for_status()

#             data = response.json()
            
#             # 1. Вывод результатов расчета
#             print("Результаты расчета:")
#             print(f"Максимальный прогиб: {max(data['solver_results']['deflection'])}")
            
#             # 2. Отображение графика
#             if 'visualization' in data and 'plot' in data['visualization']:
#                 img_data = base64.b64decode(data['visualization']['plot'])
#                 img = Image.open(BytesIO(img_data))
                
#                 plt.figure(figsize=(10, 5))
#                 plt.imshow(img)
#                 plt.axis('off')
#                 plt.title("Результаты расчета балки")
#                 plt.show()
#             else:
#                 print("График не был сгенерирован")

#     except httpx.HTTPStatusError as e:
#         print(f"Ошибка сервера ({e.response.status_code}): {e.response.text}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"Ошибка: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     asyncio.run(run_analysis())