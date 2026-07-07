import time

import requests

CHATBOT_URL = "http://localhost:8000/hospital-rag-agent"

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
outputs = [requests.post(CHATBOT_URL, json=data) for data in request_bodies]
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
