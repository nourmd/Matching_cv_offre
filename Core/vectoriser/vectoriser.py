from sentence_transformers import SentenceTransformer
import numpy as np
class Vectorizer:
    def __init__(self, model_name="intfloat/multilingual-e5-base"):
        self.model = SentenceTransformer(model_name)

    def embed_candidate(self, application: JobApplication) -> np.ndarray:
        texts = [
            f"CANDIDATE ID: {application.id}",
            "SKILLS: " + ", ".join([s.name for s in application.skills.all()]),
            "EDUCATION: " + " | ".join([e.degree or "" for e in application.educations.all()]),
            "EXPERIENCES: " + " | ".join([f"{exp.title or ''} - {exp.description or ''}" for exp in application.experiences.all()]),
            "LANGUAGES: " + " | ".join([f"{l.language or ''} ({l.level or ''})" for l in application.languages.all()]),
            "CERTIFICATIONS: " + ", ".join([c.certification or "" for c in application.certifications.all()])
        ]
        full_text = "\n".join(texts)
        return self.model.encode(full_text, convert_to_numpy=True, normalize_embeddings=True)

    def embed_offer(self, parsed_offer: ParsedJobOffer) -> np.ndarray:
        texts = [
            parsed_offer.company or "",
            parsed_offer.location or "",
            parsed_offer.contract_type or ""
        ]
        texts.extend([str(skill) for skill in parsed_offer.required_skills or []])
        texts.extend([f"{l.get('name','')} {l.get('level','')}" for l in (parsed_offer.required_languages or [])])
        texts.append(parsed_offer.required_education or "")
        texts.append(parsed_offer.required_experience or "")
        texts.extend([str(cert) for cert in parsed_offer.required_certifications or []])
        full_text = " ".join(texts)
        return self.model.encode(full_text, convert_to_numpy=True, normalize_embeddings=False)
