import json
import ollama
from typing import Optional

def get_job_prompt(job_text: str) -> str:
    return f"""
You are an expert job offer parser.

Your tasks are:
1. Extract structured information.

Here is the raw text extracted from a job offer:
\"\"\"{job_text}\"\"\" 

Now extract the following structured information in valid JSON format:

{{
  "title": "",
  "company": "",
  "location": "",
  "contract_type": "",
  "required_skills": [],
  "required_languages": [
    {{
      "language": "",
      "level": ""
    }}
  ],
  "required_education": "",
  "required_experience": "",
  "required_certifications": []
}}

Rules:
- Output only valid JSON.
- Do NOT include any explanatory text.
- If certain fields are missing, return empty strings or empty lists.
"""

class JobParser:
    def __init__(self, model_name="mistral"):
        self.client = ollama.Client(host="http://localhost:11434")
        self.model_name = model_name

    def parse_job(self, text: str) -> Optional[dict]:
        try:
            prompt = get_job_prompt(text)
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                options={"temperature":0.1, "num_ctx":4096}
            )
            return json.loads(response['response'])
        except Exception as e:
            print(f" Parsing Job erreur: {e}")
            return None
