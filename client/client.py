import httpx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

async def run_analysis():
    async with httpx.AsyncClient() as client:
        # Запрос к API Gateway
        response = await client.post(
            "http://localhost:8000/full-analysis",
            json={"k": 1, "N": 21, "method": "finite_difference"}
        )
        
        if response.status_code != 200:
            print(f"Ошибка: {response.text}")
            return

        data = response.json()
        
        # Декодирование графика
        img_data = base64.b64decode(data["plot"])
        img = BytesIO(img_data)
        
        # Отображение результатов
        plt.figure(figsize=(10, 5))
        plt.imshow(plt.imread(img))
        plt.axis('off')
        plt.title("Результаты расчета балки")
        plt.show()

        print("Результаты расчета:")
        print(f"Максимальный прогиб: {max(data['calculation']['deflection'])}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_analysis())