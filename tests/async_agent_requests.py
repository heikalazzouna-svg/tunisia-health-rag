import asyncio
import time

import httpx

CHATBOT_URL = "http://localhost:8000/hospital-rag-agent"


async def make_async_post(url, data):
    timeout = httpx.Timeout(timeout=120)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, timeout=timeout)
        return response


async def make_bulk_requests(url, data):
    tasks = [make_async_post(url, payload) for payload in data]
    responses = await asyncio.gather(*tasks)
    outputs = [r.json()["output"] for r in responses]
    return outputs


questions = [
    "What is the current wait time at Hopital Charles Nicolle?",
    "Which hospital has the shortest wait time?",
    "At which hospitals are patients complaining about billing and insurance issues?",
    "What is the average duration in days for emergency visits?",
    "What are patients saying about the nursing staff at Hopital Sahloul?",
    "What was the total billing amount charged to each payer for 2023?",
    "What is the average billing amount for CNAM visits?",
    "How many patients has Dr. Aymen Chaabane treated?",
    "Which physician has the lowest average visit duration in days?",
    "How many visits are open and what is their average duration in days?",
    "Have any patients complained about noise?",
    "How much was billed for patient 789's stay?",
    "Which physician has billed the most to STAR Assurances?",
    "Which governorate had the largest percent increase in CNAM visits from 2022 to 2023?",
]

request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = asyncio.run(make_bulk_requests(CHATBOT_URL, request_bodies))
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
