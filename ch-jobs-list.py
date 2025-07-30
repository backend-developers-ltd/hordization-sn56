import asyncio

import httpx

URL = "https://api.gradients.io/auditing/tasks"

async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(URL)
        response.raise_for_status()

        jobs = response.json()

        for job in jobs:
            if job.get("test_data") is None:
                continue
            print(
                f"{job['task_id']}   "
                f"{job['task_type']:16}   "
                f"{job['model_id']}   "
            )

if __name__ == "__main__":
    asyncio.run(main())
