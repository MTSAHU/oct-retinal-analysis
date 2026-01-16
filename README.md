# Medical Decision Support System

A backend system for **vision-assisted medical decision support**, combining
deep learning–based image analysis with large language model (LLM)–based
symptom reasoning.

> ⚠️ This system is designed for **decision support and guidance only** and
> does **not** provide medical diagnosis or prescriptions.

---

## 🔍 Current Features (Phase 1)

### ✅ Eye Disease Analysis
- Vision model: **EfficientNetV2-B3**
- Input: Retinal image (RGB, 300×300)
- Output: Predicted eye condition with confidence

### ✅ Symptom-Based Medical Reasoning
- LLM-powered reasoning via **OpenRouter (Mistral Small 3.2 – 24B)**
- Structured JSON output:
  - Possible conditions
  - Risk level (Low / Medium / High)
  - Recommended next steps
  - Follow-up questions
  - Medical disclaimer

---

## 🧠 System Architecture

User Input (Image / Symptoms)
↓
EfficientNetV2-B3 (Vision Model)
↓
Prediction + Confidence
↓
LLM Reasoner (Mistral 24B via OpenRouter)
↓
Structured Medical Guidance (JSON)


---

## 🛠 Tech Stack

- **Backend**: FastAPI
- **Vision Model**: TensorFlow / Keras (EfficientNetV2-B3)
- **Language Model**: Mistral Small 3.2 (24B) via OpenRouter
- **Environment**: Python 3.10+

---

## 🚀 Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt

    2. Set environment variables

   Create a .env file in the project root:
   OPENROUTER_API_KEY=your_api_key_here

   3. Place model weights

   Trained model weights are not included in this repository due to size constraints.

   Place the eye model file at:
   Backend/models/eye_efficientnet_v2_b3.keras

   Model weights can be distributed via GitHub Releases or external storage.
   
   Download the trained model from GitHub Releases:
    https://github.com/MTSAHU/oct-retinal-analysis/releases
      (Use release v2.0)


   4. Start the server
   uvicorn Backend.main:app --reload


   API documentation will be available at:

   http://127.0.0.1:8000/docs
