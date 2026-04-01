import os
import json
import logging
import PyPDF2
from PIL import Image
from pdf2image import convert_from_path
from docx import Document
import ollama
import pytesseract


logging.basicConfig(level=logging.INFO)
client = ollama.Client(host='http://localhost:11434')

def extract_text(file_path: str):
    try:
        file_path = os.path.normpath(file_path)
        text = ""

        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])

        elif file_path.lower().endswith(('.docx', '.doc')):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])

        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

        return text  

    except Exception as e:
        logging.warning(f"[EXTRACTION ERROR] {file_path}: {e}")
        return None
def get_cv_prompt(cv_text: str) -> str:
    return f"""
##SYSTEM ROLE:    
You are an expert AI assistant specialized in parsing resumes 
and extracting structured data for a recruitment system.

Here is the raw text extracted from a resume:
\"\"\"{cv_text}\"\"\"

##INSTRUCTIONS
- Extract the structured information and return it in **strict valid JSON format only**.
- Do NOT add any explanations, comments, or markdown formatting.
- If a field is missing, **return an empty string "" or empty list [] accordingly**.
- Include jobs, internships, and relevant projects in the experience section.
- Maintain the exact key order as shown below.
- Output MUST be parsable JSON only.

## RULES:
- Always return the result in JSON format.
- Always include **all the following keys**,
 even if the corresponding section is empty.
- DO NOT invent, guess, or hallucinate missing data.
- Only extract what is explicitly present in the CV.
- Validate and confirm if each section is present or not 
in the final result.
- Do not add extra information or text outside the JSON block.

## SECTIONS TO RETURN (JSON FORMAT):
{{
  "name": "",
  "email": "",
  "phone": "",
  "linkedin": "",
  "github": "",
  "other": [{{"type": "", "link": ""}}],
  "address": "",
  "skills": [],
  "languages": [{{"language": "", "level": ""}}],
  "education": [{{"degree": "", "institution": "", "start_date": "", "end_date": ""}}],
  "experience": [{{"title": "", "company": "", "start_date": "", "end_date": "", "description": "", "type": ""}}],
  "years_of_experience": "",
  "certifications": [{{"certification": "", "obtention_date": ""}}]
}}

##EXAMPLE OUTPUT (for reference only):
{{
  "name": "Sami Ben Ahmed",
  "email": "sami.ahmed@gmail.com",
  "phone": "+216 99 123 456",
  "linkedin": "",
  "github": "",
  "other": [],
  "address": "Tunis, Tunisia",
  "skills": ["Python", "Data Science", "SQL", "TensorFlow"],
  "languages": [
    {{"language": "French", "level": "Professional"}},
    {{"language": "English", "level": "Fluent"}}
  ],
  "education": [
    {{
      "degree": "Engineering Degree in Data Science",
      "institution": "INSAT",
      "start_date": "2019",
      "end_date": "2022"
    }}
  ],
  "experience": [
    {{
      "title": "Data Analyst Intern",
      "company": "BIAT",
      "start_date": "June 2022",
      "end_date": "August 2022",
      "description": "Worked on client churn prediction using classification models.",
      "type": "Internship"
    }}
  ],
  "years_of_experience": "1",
  "certifications": []
}}


"""
def parse_cv(file_path: str):
    text = extract_text(file_path)
    if not text:
        return None
    try:
        prompt = get_cv_prompt(text)
        response = client.generate(
            model="mistral",
            prompt=prompt,
            format="json",
            options={"temperature": 0.1, "num_ctx": 4096}
        )
        return json.loads(response.get('response', '{}'))
    except Exception as e:
        logging.warning(f"[PARSING ERROR] {file_path}: {e}")
        return None
