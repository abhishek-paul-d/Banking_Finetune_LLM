"""Locust load test against the FastAPI gateway: locust -f loadtest/locustfile.py --host http://localhost:8080"""
import json
import random

from locust import HttpUser, between, task

MESSAGES = [json.loads(l)["prompt"][1]["content"] for l in open("data/test.jsonl", encoding="utf-8")][:500]


class TriageUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def triage(self):
        self.client.post("/triage", json={"message": random.choice(MESSAGES)})
