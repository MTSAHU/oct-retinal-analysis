import json
import os
import requests
from dotenv import load_dotenv
from reasoning.schemas import SymptomInput

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "mistralai/mistral-small-24b-instruct-2501"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://medical-decision-support-api",
    "X-Title": "Medical Decision Support API"
}

SYSTEM_PROMPT = (
    "You are a medical decision-support assistant.\n"
    "You must respond ONLY in valid JSON.\n"
    "Do NOT provide a diagnosis.\n"
    "Do NOT prescribe medications.\n"
    "Provide general medical guidance only.\n\n"
    "The JSON response must have exactly these fields:\n"
    "{\n"
    "  \"possible_conditions\": [string],\n"
    "  \"risk_level\": \"Low | Medium | High\",\n"
    "  \"next_steps\": [string],\n"
    "  \"follow_up_questions\": [string],\n"
    "  \"disclaimer\": string\n"
    "}\n"
)

def build_llm_prompt(data: SymptomInput) -> str:
    return f"""
Patient Information:
Age: {data.age}
Gender: {data.gender}
Location: {data.location}

Primary Symptom: {data.primary_symptom}
Additional Symptoms: {", ".join(data.additional_symptoms or [])}
Duration: {data.duration}
Severity (1-5): {data.severity}

Family History: {data.family_history}
Chronic Condition: {data.chronic_condition}
Smoking/Alcohol: {data.smoking_or_alcohol}

Tasks:
1. List possible medical conditions (not diagnosis)
2. Assess risk level (Low / Medium / High)
3. Recommend next steps
4. Ask up to 2 follow-up questions if necessary
5. Include a clear disclaimer

Return the response strictly in JSON format as specified.
"""

def call_llm(prompt: str) -> dict:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 600
    }

    resp = requests.post(
        OPENROUTER_ENDPOINT,
        headers=HEADERS,
        json=payload,
        timeout=90
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error: {resp.status_code} - {resp.text}"
        )

    raw_text = resp.json()["choices"][0]["message"]["content"].strip()

    # Remove markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "possible_conditions": [],
            "risk_level": "Unknown",
            "next_steps": [],
            "follow_up_questions": [],
            "disclaimer": raw_text
        }
